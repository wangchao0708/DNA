import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import csv
import json
import gc
import time
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Key workflow:
# 1. load images from source-model folders,
# 2. encode images and prompts with each candidate SD1 model,
# 3. compute denoising MSE profiles over selected timesteps and noise samples,
# 4. save the per-noise MSE table for the final attribution evaluation.

IMAGE_ROOT = os.environ.get('DNA_IMAGE_ROOT', '')
SOURCE_FOLDERS = ['SD1.1', 'SD1.2', 'SD1.3', 'SD1.4', 'SD1.5']
MODEL_IDS = {
    'SD1.1': 'CompVis/stable-diffusion-v1-1',
    'SD1.2': 'CompVis/stable-diffusion-v1-2',
    'SD1.3': 'CompVis/stable-diffusion-v1-3',
    'SD1.4': 'CompVis/stable-diffusion-v1-4',
    'SD1.5': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
}
NUM_IMAGES = None
NUM_NOISE = None
TIMESTEPS = []
NOISE_SEED = None
BATCH_SIZE = 100
IMAGE_SIZE = 512
DEVICE = 'cuda'
DTYPE = torch.float16
USE_BLIP = False

RESULTS_CSV = 'DNA_SD1_results.csv'
SUMMARY_JSON = 'DNA_SD1_summary.json'
BLIP_CACHE = 'DNA_SD1_blip_captions.json'



def validate_runtime_config():
    missing = []
    if not IMAGE_ROOT:
        missing.append('IMAGE_ROOT or DNA_IMAGE_ROOT')
    if NUM_NOISE is None:
        missing.append('NUM_NOISE')
    if NOISE_SEED is None:
        missing.append('NOISE_SEED')
    if not TIMESTEPS:
        missing.append('TIMESTEPS')
    if missing:
        joined = ', '.join(missing)
        raise ValueError(
            f"Please configure the required runtime parameters before running this script: {joined}"
        )


def load_image_pil(path, size=512):
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def load_image_tensor(path, size=512):
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img)


def generate_captions(image_paths, device='cuda'):
    # Optional prompt construction: reuse cached BLIP-2 captions when available.
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
def encode_images_to_latents(vae, image_paths, batch_size, device, dtype, scaling_factor):
    """Encode images into latents and keep the result on GPU."""
    all_latents = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = torch.stack([load_image_tensor(p, IMAGE_SIZE) for p in batch_paths])
        batch = batch.to(device, dtype=dtype)
        posterior = vae.encode(batch).latent_dist
        latents = posterior.mean * scaling_factor  # [B, 4, 64, 64], kept on GPU
        all_latents.append(latents)
        del batch, posterior
    return torch.cat(all_latents, dim=0)  # [N, 4, 64, 64], GPU


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts, device, dtype, batch_size=32, max_length=77):
    """Batch-encode prompts into text embeddings and keep the result on GPU."""
    all_embs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokens = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        emb = text_encoder(tokens.input_ids.to(device))[0]  # [B, 77, 768], kept on GPU
        all_embs.append(emb)
        del tokens
    return torch.cat(all_embs, dim=0)  # [N, 77, 768], GPU


@torch.no_grad()
def compute_mse_multi_timestep(z0_batch, text_emb_batch, unet, scheduler,
                                timesteps, fixed_noises, device, dtype):
    """
    Return MSE for each noise sample and timestep without averaging over noise.
    Returns: [B, NUM_NOISE, len(timesteps)] as a CPU tensor.
    """
    B = z0_batch.shape[0]
    N_noise = len(fixed_noises)
    z0 = z0_batch
    text_emb = text_emb_batch

    mse_per_noise_t = torch.zeros(B, N_noise, len(timesteps))  # CPU

    for ti, t in enumerate(timesteps):
        alpha_t = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_t).sqrt()

        for ni in range(N_noise):
            # noise = fixed_noises[ni].expand(B, -1, -1, -1)





            
            noise = torch.randn(B, 4, z0.shape[2], z0.shape[3], device=device, dtype=dtype)

            z_t = sqrt_alpha * z0 + sqrt_one_minus_alpha * noise
            noise_pred = unet(z_t, t, encoder_hidden_states=text_emb).sample

            mse = (noise - noise_pred).pow(2).mean(dim=[1, 2, 3]).cpu()
            mse_per_noise_t[:, ni, ti] = mse

            del z_t, noise_pred, mse

    return mse_per_noise_t  # [B, NUM_NOISE, len(timesteps)], CPU


def main():
    validate_runtime_config()
    from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    start_time = time.time()

    # Prepare the fixed-noise bank used by all candidate models in this family.
    print("=" * 70)
    print("  Improved MSE attribution: BLIP captions, multiple timesteps, aggregation, and per-noise MSE export.")
    print("=" * 70)
    print(f"  USE_BLIP={USE_BLIP}, timesteps={TIMESTEPS}")
    print(f"  noise samples={NUM_NOISE}, NOISE_SEED={NOISE_SEED}, batch size={BATCH_SIZE}")

    # ====== Build the fixed-noise bank directly on GPU======
    print(f"\n[0/5] Build the fixed-noise bank (seed={NOISE_SEED})...")
    latent_h = IMAGE_SIZE // 8  # 512 -> 64
    rng = torch.Generator()
    rng.manual_seed(NOISE_SEED)
    fixed_noises = [
        torch.randn(1, 4, latent_h, latent_h, generator=rng).to(DEVICE, dtype=DTYPE)
        for _ in range(NUM_NOISE)
    ]
    print(f"  Generated {NUM_NOISE} fixed noise samples, shape={fixed_noises[0].shape}, device={fixed_noises[0].device}")

    # ====== Collect all image paths ======
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

    # ====== Generating captions (run once with cache support)======
    captions = {}
    if USE_BLIP:
        print("\n[1/5] Generating BLIP captions...")
        all_image_paths = []
        for paths in all_paths.values():
            all_image_paths.extend(paths)
        captions = generate_captions(all_image_paths, DEVICE)
    else:
        print("\n[1/5] Skipping BLIP and using empty prompts")

    # ====== Prepare the CSV file with a noise_idx column======
    csv_cols = ['source', 'img_idx', 'model', 'noise_idx']
    for t in TIMESTEPS:
        csv_cols.append(f'mse_t{t}')
    with open(RESULTS_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(csv_cols)

    # ====== Store all intermediate results ======
    # (source, idx, model) -> avg_mse, averaged over noise and timesteps for Method A.
    all_results = {}
    # (source, idx, model) -> np array of shape [len(timesteps)] (averaged over noise for accuracy methods B/C/D)
    all_mse_profiles = {}
    # Record the number of images per source folder for later accuracy computation
    folder_n = {}

    # ====== For each candidate: load components, encode, predict, and release memory ======
    print(f"\n[2-4/5] Load each candidate, encode inputs, and compute MSE...")

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n{'=' * 70}")
        print(f"  model: {model_name}  ({model_id})")
        print(f"{'=' * 70}")
        model_start = time.time()

        # ---- Load the candidate VAE ----
        print(f"  [component 1/3] Loading VAE...")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').to(DEVICE, dtype=DTYPE)
        vae.eval()
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)

        # ---- Encode all images with the candidate VAE -> latent (GPU)----
        print(f"  Encoding images into latent space...")
        model_latents = {}
        for folder, paths in all_paths.items():
            latents = encode_images_to_latents(vae, paths, BATCH_SIZE, DEVICE, DTYPE, scaling_factor)
            model_latents[folder] = latents  # GPU
            folder_n[folder] = latents.shape[0]
            print(f"    {folder}: latent shape={latents.shape}, device={latents.device}")

        del vae
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Load the candidate tokenizer and TextEncoder ----
        print(f"  [component 2/3] Loading TextEncoder...")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder').to(DEVICE, dtype=DTYPE)
        text_encoder.eval()

        # ---- Encode all prompts with the candidate TextEncoder -> text_emb (GPU)----
        print(f"  Encoding text embeddings...")
        model_text_embs = {}
        for folder, paths in all_paths.items():
            prompts = [captions[p] if (USE_BLIP and p in captions) else "" for p in paths]
            text_embs = encode_prompts(tokenizer, text_encoder, prompts, DEVICE, DTYPE,
                                       batch_size=BATCH_SIZE)
            model_text_embs[folder] = text_embs  # GPU
            print(f"    {folder}: text_emb shape={text_embs.shape}, device={text_embs.device}")

        del text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Load the candidate UNet and scheduler ----
        print(f"  [component 3/3] Loading UNet + scheduler...")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet').to(DEVICE, dtype=DTYPE)
        unet.eval()
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')

        # ---- Compute MSE for each source folder ----
        for source in SOURCE_FOLDERS:
            if source not in model_latents:
                continue
            latents   = model_latents[source]    # GPU
            text_embs = model_text_embs[source]  # GPU
            n = latents.shape[0]

            pbar = tqdm(range(0, n, BATCH_SIZE), desc=f'    {source}->{model_name}', ncols=80)
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, n)
                batch_z0  = latents[batch_start:batch_end]    # GPU slice, no copy
                batch_emb = text_embs[batch_start:batch_end]  # GPU slice, no copy

                mse_per_noise_t = compute_mse_multi_timestep(
                    batch_z0, batch_emb, unet, scheduler,
                    TIMESTEPS, fixed_noises,
                    DEVICE, DTYPE
                )  # returns a CPU tensor [B, NUM_NOISE, len(TIMESTEPS)]

                rows = []
                for j in range(mse_per_noise_t.shape[0]):
                    idx = batch_start + j
                    noise_arr = mse_per_noise_t[j].numpy()  # [NUM_NOISE, len(TIMESTEPS)]

                    # Noise-averaged profile used for accuracy computation, matching the original logic
                    mse_arr = noise_arr.mean(axis=0)  # [len(TIMESTEPS)]
                    avg_mse = float(mse_arr.mean())

                    all_results[(source, idx, model_name)] = avg_mse
                    all_mse_profiles[(source, idx, model_name)] = mse_arr

                    # Write one CSV row per noise sample
                    for ni in range(noise_arr.shape[0]):
                        row = [source, idx, model_name, ni]
                        for ti in range(len(TIMESTEPS)):
                            row.append(f"{noise_arr[ni, ti]:.10f}")
                        rows.append(row)

                with open(RESULTS_CSV, 'a', newline='') as f:
                    csv.writer(f).writerows(rows)

                del mse_per_noise_t

        elapsed = time.time() - model_start
        print(f"  model {model_name} completed, elapsed {elapsed:.1f}s")

        # ---- Release the candidate components and intermediate tensors ----
        del unet, scheduler, model_latents, model_text_embs
        gc.collect()
        torch.cuda.empty_cache()

    # ====== Compute accuracy from the noise-averaged MSE profiles ======
    print(f"\n[5/5] Computing attribution accuracy...")

    # === Method A: average MSE over all timesteps ===
    correct_avg = 0
    total = 0
    confusion_avg = {s: {m: 0 for m in SOURCE_FOLDERS} for s in SOURCE_FOLDERS}

    for source in SOURCE_FOLDERS:
        if source not in folder_n:
            continue
        n = folder_n[source]
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

    # === Method B: per-timestep attribution ===
    per_t_correct = {t: 0 for t in TIMESTEPS}
    per_t_total   = {t: 0 for t in TIMESTEPS}

    for source in SOURCE_FOLDERS:
        if source not in folder_n:
            continue
        n = folder_n[source]
        for idx in range(n):
            for ti, t in enumerate(TIMESTEPS):
                scores = {}
                for model_name in MODEL_IDS:
                    key = (source, idx, model_name)
                    if key in all_mse_profiles:
                        scores[model_name] = float(all_mse_profiles[key][ti])
                if scores:
                    pred = min(scores, key=scores.get)
                    per_t_total[t] += 1
                    if pred == source:
                        per_t_correct[t] += 1

    # === Method C: best single timestep ===
    best_t     = max(TIMESTEPS, key=lambda t: per_t_correct[t] / per_t_total[t] if per_t_total[t] > 0 else 0)
    acc_best_t = per_t_correct[best_t] / per_t_total[best_t] if per_t_total[best_t] > 0 else 0

    # === Method D: weighted MSE with stronger low-noise timestep weights ===
    weight_schemes = {
        'uniform':     np.ones(len(TIMESTEPS)),
        'low_t_heavy': np.array([1.0 / (t + 1) for t in TIMESTEPS]),
        'low_only':    np.array([1.0 if t <= 200 else 0.0 for t in TIMESTEPS]),
        'mid_only':    np.array([1.0 if 100 <= t <= 500 else 0.0 for t in TIMESTEPS]),
        'high_only':   np.array([1.0 if t >= 500 else 0.0 for t in TIMESTEPS]),
    }

    best_scheme_name = 'uniform'
    best_scheme_acc  = acc_avg
    scheme_results   = {}

    for scheme_name, weights in weight_schemes.items():
        if weights.sum() < 1e-8:
            continue
        weights   = weights / weights.sum()
        s_correct = 0
        s_total   = 0
        for source in SOURCE_FOLDERS:
            if source not in folder_n:
                continue
            n = folder_n[source]
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
            best_scheme_acc  = s_acc
            best_scheme_name = scheme_name

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  Summary (USE_BLIP={USE_BLIP}, NOISE_SEED={NOISE_SEED})")
    print(f"{'=' * 70}")

    print(f"\n  Method A - average MSE over all timesteps: {acc_avg:.4f} ({correct_avg}/{total})")

    print(f"\n  Method B - per-timestep accuracy:")
    for t in TIMESTEPS:
        if per_t_total[t] > 0:
            acc_t  = per_t_correct[t] / per_t_total[t]
            marker = " <- BEST" if t == best_t else ""
            print(f"    t={t:4d}: {acc_t:.4f}{marker}")

    print(f"\n  Method C - best single timestep: t={best_t}, acc={acc_best_t:.4f}")

    print(f"\n  Method D - weighted schemes:")
    for scheme_name, acc_s in scheme_results.items():
        marker = " <- BEST" if scheme_name == best_scheme_name else ""
        print(f"    {scheme_name:>15}: {acc_s:.4f}{marker}")

    print(f"\n  Confusion matrix (average MSE over all timesteps):")
    header = f"{'':>10}" + "".join(f"{m:>10}" for m in SOURCE_FOLDERS)
    print(f"  {header}")
    for s in SOURCE_FOLDERS:
        row = f"{s:>10}" + "".join(f"{confusion_avg[s][m]:>10}" for m in SOURCE_FOLDERS)
        print(f"  {row}")

    # Save
    summary = {
        'use_blip':   USE_BLIP,
        'noise_seed': NOISE_SEED,
        'accuracy_avg_mse':        acc_avg,
        'accuracy_best_single_t':  {'t': best_t, 'acc': acc_best_t},
        'accuracy_per_timestep': {
            str(t): per_t_correct[t] / per_t_total[t] if per_t_total[t] > 0 else 0
            for t in TIMESTEPS
        },
        'accuracy_weighted_schemes': scheme_results,
        'best_weighted_scheme':      {'name': best_scheme_name, 'acc': best_scheme_acc},
        'confusion_matrix_avg':      confusion_avg,
        'total': total,
        'config': {
            'timesteps':  TIMESTEPS,
            'num_noise':  NUM_NOISE,
            'noise_seed': NOISE_SEED,
            'num_images': NUM_IMAGES,
            'batch_size': BATCH_SIZE,
        },
        'total_time': time.time() - start_time,
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total elapsed: {(time.time() - start_time) / 60:.1f} min")
    print(f"  Outputs: {RESULTS_CSV}, {SUMMARY_JSON}")


if __name__ == '__main__':
    main()
