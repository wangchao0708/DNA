import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import csv
import json
import gc
import time
from tqdm import tqdm

IMAGE_ROOT = os.environ.get('DNA_IMAGE_ROOT', '')
SOURCE_FOLDERS = ['SD3-M', 'SD3.5-L', 'SD3.5-LT', 'SD3.5-M']
MODEL_IDS = {
    'SD3-M': 'stabilityai/stable-diffusion-3-medium-diffusers',
    'SD3.5-L': 'stabilityai/stable-diffusion-3.5-large',
    'SD3.5-LT': 'stabilityai/stable-diffusion-3.5-large-turbo',
    'SD3.5-M': 'stabilityai/stable-diffusion-3.5-medium',
}
NUM_IMAGES = None
NUM_NOISE = None
SIGMAS = []
NOISE_SEED = None
BATCH_SIZE = 25
IMAGE_SIZE = 1024
LATENT_CHANNELS = 16
DEVICE = 'cuda'
DTYPE = torch.bfloat16
USE_BLIP = False
RESULTS_CSV = 'DNA_SD3_results.csv'
SUMMARY_JSON = 'DNA_SD3_summary.json'
BLIP_CACHE = 'DNA_SD3_blip_captions.json'



def validate_runtime_config():
    missing = []
    if not IMAGE_ROOT:
        missing.append('IMAGE_ROOT or DNA_IMAGE_ROOT')
    if NUM_NOISE is None:
        missing.append('NUM_NOISE')
    if NOISE_SEED is None:
        missing.append('NOISE_SEED')
    if not SIGMAS:
        missing.append('SIGMAS')
    if missing:
        joined = ', '.join(missing)
        raise ValueError(
            f"Please configure the required runtime parameters before running this script: {joined}"
        )


def load_image_pil(path, size=1024):
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def load_image_tensor(path, size=1024):
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img)


def generate_captions(image_paths, device='cuda'):
    if os.path.exists(BLIP_CACHE):
        print(f"  Found caption cache {BLIP_CACHE}, loading it directly...")
        with open(BLIP_CACHE, 'r') as f:
            captions = json.load(f)
        missing = [p for p in image_paths if p not in captions]
        if not missing:
            print(f"  Cache hit with {len(captions)} captions; skipping BLIP inference")
            return captions
        else:
            print(f"  Caption cache is incomplete; missing {len(missing)} entries; generating the missing captions...")
            image_paths = missing
    else:
        captions = {}

    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print("  Loading the BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    print("  Generating captions...")
    for path in tqdm(image_paths, ncols=80):
        img = load_image_pil(path)
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        captions[path] = caption

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    with open(BLIP_CACHE, 'w') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(captions)} captions to {BLIP_CACHE}")

    return captions


@torch.no_grad()
def encode_images_to_latents(vae, image_paths, batch_size, device, dtype):
    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, 'shift_factor', 0.0)
    print(f"    VAE scaling_factor={scaling_factor}, shift_factor={shift_factor}")

    all_latents = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = torch.stack([load_image_tensor(p, IMAGE_SIZE) for p in batch_paths])
        batch = batch.to(device, dtype=dtype)
        posterior = vae.encode(batch).latent_dist
        latents = (posterior.mean - shift_factor) * scaling_factor
        all_latents.append(latents.cpu())
        del batch, posterior, latents
    return torch.cat(all_latents, dim=0)


@torch.no_grad()
def encode_prompts_sd3(pipe, prompts, device):
    all_prompt_embeds = []
    all_pooled_embeds = []

    for prompt in tqdm(prompts, desc="    encoding prompts", ncols=80):
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=256,
        )
        all_prompt_embeds.append(prompt_embeds.cpu())
        all_pooled_embeds.append(pooled_prompt_embeds.cpu())

    return torch.cat(all_prompt_embeds, dim=0), torch.cat(all_pooled_embeds, dim=0)


@torch.no_grad()
def compute_mse_multi_sigma(z0_batch, prompt_emb_batch, pooled_emb_batch,
                             transformer, sigmas, fixed_noises, device, dtype):
    """
    Returns: mse_per_noise_sigma [B, NUM_NOISE, len(sigmas)].
    """
    B = z0_batch.shape[0]
    z0 = z0_batch.to(device, dtype=dtype)
    prompt_emb = prompt_emb_batch.to(device, dtype=dtype)
    pooled_emb = pooled_emb_batch.to(device, dtype=dtype)

    mse_per_noise_sigma = torch.zeros(B, len(fixed_noises), len(sigmas))

    for si, sigma in enumerate(sigmas):
        for ni in range(len(fixed_noises)):
            # noise = fixed_noises[ni].to(device, dtype=dtype).expand(B, -1, -1, -1)
            noise = torch.randn(B, LATENT_CHANNELS, z0.shape[2], z0.shape[3], device=device, dtype=dtype)





            z_t = (1.0 - sigma) * z0 + sigma * noise
            v_target = noise - z0

            timestep = torch.tensor([sigma * 1000.0] * B, device=device, dtype=dtype)
            v_pred = transformer(
                hidden_states=z_t,
                timestep=timestep,
                encoder_hidden_states=prompt_emb,
                pooled_projections=pooled_emb,
                return_dict=False,
            )[0]

            mse = (v_target - v_pred).pow(2).mean(dim=[1, 2, 3]).cpu()
            mse_per_noise_sigma[:, ni, si] = mse

            del z_t, v_target, v_pred, mse, timestep

    return mse_per_noise_sigma


def main():
    validate_runtime_config()
    from diffusers import StableDiffusion3Pipeline

    start_time = time.time()

    print("=" * 70)
    print("  Improved MSE attribution: flow matching, BLIP captions, multiple sigmas, and fixed noise.")
    print("  Adapted for SD3 / SD3.5; each candidate uses its own VAE and text encoder.")
    print("=" * 70)
    print(f"  USE_BLIP={USE_BLIP}, number of sigmas={len(SIGMAS)}")
    print(f"  noise samples={NUM_NOISE}, NOISE_SEED={NOISE_SEED}, batch size={BATCH_SIZE}")
    print(f"  latent channels={LATENT_CHANNELS}, dtype={DTYPE}")

    # ====== [0/4] Build the fixed-noise bank ======
    print(f"\n[0/4] Build the fixed-noise bank (seed={NOISE_SEED})...")
    latent_h = IMAGE_SIZE // 8
    rng = torch.Generator()
    rng.manual_seed(NOISE_SEED)
    fixed_noises = [
        torch.randn(1, LATENT_CHANNELS, latent_h, latent_h, generator=rng)
        for _ in range(NUM_NOISE)
    ]
    print(f"  Generated {NUM_NOISE} fixed noise samples, shape={fixed_noises[0].shape}")

    # ====== Collect image paths ======
    all_paths = {}
    for folder in SOURCE_FOLDERS:
        folder_path = os.path.join(IMAGE_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: {folder_path} does not exist")
            continue
        paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.png')
        ])[:NUM_IMAGES]
        if paths:
            all_paths[folder] = paths
            print(f"  {folder}: {len(paths)} images")

    # ====== [1/4] BLIP captions ======
    captions = {}
    if USE_BLIP:
        print("\n[1/5] Generating BLIP captions...")
        all_image_paths = []
        for paths in all_paths.values():
            all_image_paths.extend(paths)
        captions = generate_captions(all_image_paths, DEVICE)
    else:
        print("\n[1/5] Skipping BLIP and using empty prompts")

    # ====== Prepare the CSV file ======
    csv_cols = ['source', 'img_idx', 'model', 'noise_idx']
    for s in SIGMAS:
        csv_cols.append(f'mse_t{int(s*1000)}')
    with open(RESULTS_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(csv_cols)

    # ====== [2/4] per-candidate: Loading full pipeline -> encode -> compute MSE -> release ======
    print(f"\n[2/4] Load each full pipeline, encode inputs, and compute MSE...")

    all_results = {}
    all_mse_profiles = {}

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n  {'-' * 55}")
        print(f"  Loading full pipeline: {model_name} ({model_id})")

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=DTYPE
        ).to(DEVICE)

        print(f"  Encoding images into latent space (using {model_name} VAE)...")
        model_latents = {}
        for folder, paths in all_paths.items():
            print(f"    Encoding {folder}...", end=' ', flush=True)
            latents = encode_images_to_latents(
                pipe.vae, paths, BATCH_SIZE, DEVICE, DTYPE
            )
            model_latents[folder] = latents
            print(f"shape={latents.shape}")

        print(f"  Encoding text embeddings (using {model_name} text encoder)...")
        model_prompt_embs = {}
        model_pooled_embs = {}
        for folder, paths in all_paths.items():
            prompts = [captions.get(p, "") if USE_BLIP else "" for p in paths]
            prompt_embs, pooled_embs = encode_prompts_sd3(pipe, prompts, DEVICE)
            model_prompt_embs[folder] = prompt_embs
            model_pooled_embs[folder] = pooled_embs
            print(f"    {folder}: prompt_embs={prompt_embs.shape}, pooled={pooled_embs.shape}")

        pipe.vae.cpu()
        if pipe.text_encoder is not None:
            pipe.text_encoder.cpu()
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.cpu()
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Moved the VAE and text encoder to CPU; only the transformer remains on GPU")

        transformer = pipe.transformer
        transformer.eval()
        model_start = time.time()

        for source in SOURCE_FOLDERS:
            if source not in model_latents:
                continue
            latents = model_latents[source]
            prompt_embs = model_prompt_embs[source]
            pooled_embs = model_pooled_embs[source]
            n = latents.shape[0]

            pbar = tqdm(range(0, n, BATCH_SIZE),
                        desc=f'    {source}->{model_name}', ncols=80)
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, n)
                batch_z0 = latents[batch_start:batch_end]
                batch_prompt = prompt_embs[batch_start:batch_end]
                batch_pooled = pooled_embs[batch_start:batch_end]

                mse_per_noise_sigma = compute_mse_multi_sigma(
                    batch_z0, batch_prompt, batch_pooled,
                    transformer, SIGMAS, fixed_noises,
                    DEVICE, DTYPE
                )

                rows = []
                for j in range(mse_per_noise_sigma.shape[0]):
                    idx = batch_start + j
                    noise_arr = mse_per_noise_sigma[j].numpy()  # [NUM_NOISE, len(SIGMAS)]
                    mse_arr = noise_arr.mean(axis=0)            # [len(SIGMAS)]
                    avg_mse = float(mse_arr.mean())

                    all_results[(source, idx, model_name)] = avg_mse
                    all_mse_profiles[(source, idx, model_name)] = mse_arr

                    for ni in range(noise_arr.shape[0]):
                        row = [source, idx, model_name, ni]
                        for si in range(len(SIGMAS)):
                            row.append(f"{noise_arr[ni, si]:.10f}")
                        rows.append(row)

                with open(RESULTS_CSV, 'a', newline='') as f:
                    csv.writer(f).writerows(rows)

                del batch_z0, batch_prompt, batch_pooled, mse_per_noise_sigma

        elapsed = time.time() - model_start
        print(f"  model {model_name} MSE computation completed, elapsed {elapsed:.1f}s")

        del transformer, pipe, model_latents, model_prompt_embs, model_pooled_embs
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Pipeline {model_name} fully released")

    # ====== [3/4] Compute accuracy ======
    print(f"\n[3/4] Computing attribution accuracy...")

    correct_avg = 0
    total = 0
    confusion_avg = {s: {m: 0 for m in SOURCE_FOLDERS} for s in SOURCE_FOLDERS}

    for source in SOURCE_FOLDERS:
        if source not in all_paths:
            continue
        n = len(all_paths[source])
        for idx in range(n):
            scores = {}
            for model_name in MODEL_IDS:
                key = (source, idx, model_name)
                if key in all_results:
                    scores[model_name] = all_results[key]
            if not scores:
                continue
            predicted = min(scores, key=scores.get)
            confusion_avg[source][predicted] += 1
            total += 1
            if predicted == source:
                correct_avg += 1

    acc_avg = correct_avg / total if total > 0 else 0

    per_s_correct = {i: 0 for i in range(len(SIGMAS))}
    per_s_total = {i: 0 for i in range(len(SIGMAS))}

    for source in SOURCE_FOLDERS:
        if source not in all_paths:
            continue
        n = len(all_paths[source])
        for idx in range(n):
            for si in range(len(SIGMAS)):
                scores = {}
                for model_name in MODEL_IDS:
                    key = (source, idx, model_name)
                    if key in all_mse_profiles:
                        scores[model_name] = float(all_mse_profiles[key][si])
                if scores:
                    pred = min(scores, key=scores.get)
                    per_s_total[si] += 1
                    if pred == source:
                        per_s_correct[si] += 1

    best_si = max(range(len(SIGMAS)),
                  key=lambda i: per_s_correct[i] / per_s_total[i] if per_s_total[i] > 0 else 0)
    best_sigma = SIGMAS[best_si]
    acc_best_sigma = per_s_correct[best_si] / per_s_total[best_si] if per_s_total[best_si] > 0 else 0

    sigmas_arr = np.array(SIGMAS)
    weight_schemes = {
        'uniform':     np.ones(len(SIGMAS)),
        'low_s_heavy': 1.0 / (sigmas_arr + 0.01),
        'low_only':    np.where(sigmas_arr <= 0.2, 1.0, 0.0),
        'mid_only':    np.where((sigmas_arr >= 0.1) & (sigmas_arr <= 0.5), 1.0, 0.0),
        'high_only':   np.where(sigmas_arr >= 0.5, 1.0, 0.0),
    }

    best_scheme_name = 'uniform'
    best_scheme_acc = acc_avg
    scheme_results = {}

    for scheme_name, weights in weight_schemes.items():
        if weights.sum() < 1e-8:
            continue
        weights = weights / weights.sum()
        s_correct = 0
        s_total = 0
        for source in SOURCE_FOLDERS:
            if source not in all_paths:
                continue
            n = len(all_paths[source])
            for idx in range(n):
                scores = {}
                for model_name in MODEL_IDS:
                    key = (source, idx, model_name)
                    if key in all_mse_profiles:
                        scores[model_name] = float((all_mse_profiles[key] * weights).sum())
                if scores:
                    pred = min(scores, key=scores.get)
                    s_total += 1
                    if pred == source:
                        s_correct += 1
        s_acc = s_correct / s_total if s_total > 0 else 0
        scheme_results[scheme_name] = s_acc
        if s_acc > best_scheme_acc:
            best_scheme_acc = s_acc
            best_scheme_name = scheme_name

    # ====== [4/4] Print results ======
    print(f"\n{'=' * 70}")
    print(f"  Summary (USE_BLIP={USE_BLIP}, NOISE_SEED={NOISE_SEED})")
    print(f"{'=' * 70}")

    print(f"\n  Method A - average MSE over all sigmas: {acc_avg:.4f} ({correct_avg}/{total})")

    print(f"\n  Method B - per-sigma accuracy (top-10):")
    sigma_accs = [
        (SIGMAS[i], per_s_correct[i] / per_s_total[i] if per_s_total[i] > 0 else 0)
        for i in range(len(SIGMAS))
    ]
    sigma_accs_sorted = sorted(sigma_accs, key=lambda x: x[1], reverse=True)[:10]
    for sigma_val, acc_val in sigma_accs_sorted:
        marker = " <- BEST" if abs(sigma_val - best_sigma) < 1e-6 else ""
        print(f"    sigma={sigma_val:.4f} (t={sigma_val * 1000:.1f}): {acc_val:.4f}{marker}")

    print(f"\n  Method C - best single sigma: sigma={best_sigma:.4f}, acc={acc_best_sigma:.4f}")

    print(f"\n  Method D - weighted schemes:")
    for scheme_name, acc_s in scheme_results.items():
        marker = " <- BEST" if scheme_name == best_scheme_name else ""
        print(f"    {scheme_name:>15}: {acc_s:.4f}{marker}")

    print(f"\n  Confusion matrix (average MSE over all sigmas):")
    header = "{:>22}".format("true\\pred") + "".join(f"{m:>22}" for m in SOURCE_FOLDERS)
    print(f"  {header}")
    for s in SOURCE_FOLDERS:
        row = f"{s:>22}" + "".join(f"{confusion_avg[s][m]:>22}" for m in SOURCE_FOLDERS)
        print(f"  {row}")

    summary = {
        'use_blip': USE_BLIP,
        'noise_seed': NOISE_SEED,
        'per_model_encoding': True,
        'accuracy_avg_mse': acc_avg,
        'accuracy_best_single_sigma': {'sigma': best_sigma, 'acc': acc_best_sigma},
        'accuracy_per_sigma': {
            f"{SIGMAS[i]:.4f}": per_s_correct[i] / per_s_total[i] if per_s_total[i] > 0 else 0
            for i in range(len(SIGMAS))
        },
        'accuracy_weighted_schemes': scheme_results,
        'best_weighted_scheme': {'name': best_scheme_name, 'acc': best_scheme_acc},
        'confusion_matrix_avg': confusion_avg,
        'total': total,
        'config': {
            'sigmas_count': len(SIGMAS),
            'sigma_range': [SIGMAS[0], SIGMAS[-1]],
            'num_noise': NUM_NOISE,
            'noise_seed': NOISE_SEED,
            'num_images': NUM_IMAGES,
            'batch_size': BATCH_SIZE,
            'latent_channels': LATENT_CHANNELS,
            'dtype': str(DTYPE),
        },
        'total_time': time.time() - start_time,
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total elapsed: {(time.time() - start_time) / 60:.1f} min")
    print(f"  Outputs: {RESULTS_CSV}, {SUMMARY_JSON}")


if __name__ == '__main__':
    main()
