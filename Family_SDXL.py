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
# 2. encode images and prompts with each candidate SDXL-family model,
# 3. compute denoising MSE profiles over selected timesteps and noise samples,
# 4. save the per-noise MSE table for the final attribution evaluation.

IMAGE_ROOT = os.environ.get('DNA_IMAGE_ROOT', '')
SOURCE_FOLDERS = ['SDXL-0.9', 'SDXL-1.0', 'SSD', 'SV']
MODEL_IDS = {
    'SDXL-1.0': 'stabilityai/stable-diffusion-xl-base-1.0',
    'SDXL-0.9': 'stabilityai/stable-diffusion-xl-base-0.9',
    'SSD': 'segmind/SSD-1B',
    'SV': 'segmind/Segmind-Vega',
}
NUM_IMAGES = None
NUM_NOISE = None
TIMESTEPS = []
NOISE_SEED = None
BATCH_SIZE = 25
IMAGE_SIZE = 1024
DEVICE = 'cuda'
DTYPE = torch.float16
VAE_DTYPE = torch.float32
USE_BLIP = False

RESULTS_CSV = 'DNA_SDXL_results.csv'
SUMMARY_JSON = 'DNA_SDXL_summary.json'
BLIP_CACHE = 'DNA_SDXL_blip_captions.json'


def load_fp16_or_fallback(cls, model_id, subfolder, device, dtype):
    try:
        model = cls.from_pretrained(
            model_id, subfolder=subfolder,
            variant='fp16', torch_dtype=dtype
        ).to(device)
        print(f"      [OK] {subfolder}: loaded fp16 variant")
        return model
    except OSError:
        model = cls.from_pretrained(
            model_id, subfolder=subfolder,
            torch_dtype=dtype
        ).to(device)
        print(f"      [WARN] {subfolder}: fp16 variant unavailable; falling back to default weights with dtype conversion")
        return model



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
def encode_images_to_latents(vae, image_paths, batch_size, device, vae_dtype,
                              output_dtype, scaling_factor):
    all_latents = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = torch.stack([load_image_tensor(p, IMAGE_SIZE) for p in batch_paths])
        batch = batch.to(device, dtype=vae_dtype)
        posterior = vae.encode(batch).latent_dist
        latents = posterior.mean * scaling_factor
        all_latents.append(latents.to(dtype=output_dtype).cpu())
        del batch, posterior, latents
    return torch.cat(all_latents, dim=0)


@torch.no_grad()
def encode_prompts_sdxl(tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                         prompts, device, dtype, max_length=77):
    all_prompt_embeds = []
    all_pooled_embeds = []

    for prompt in prompts:
        tokens_1 = tokenizer(
            prompt, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        )
        output_1 = text_encoder(
            tokens_1.input_ids.to(device), output_hidden_states=True
        )
        hidden_states_1 = output_1.hidden_states[-2]

        tokens_2 = tokenizer_2(
            prompt, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        )
        output_2 = text_encoder_2(
            tokens_2.input_ids.to(device), output_hidden_states=True
        )
        hidden_states_2 = output_2.hidden_states[-2]
        pooled_output = output_2[0]

        prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        all_prompt_embeds.append(prompt_embeds.cpu())
        all_pooled_embeds.append(pooled_output.cpu())

    return torch.cat(all_prompt_embeds, dim=0), torch.cat(all_pooled_embeds, dim=0)


@torch.no_grad()
def compute_mse_multi_timestep(z0_batch, prompt_emb_batch, pooled_emb_batch,
                                unet, scheduler, timesteps, fixed_noises,
                                device, dtype):
    """
    Returns: mse_per_noise_t [B, NUM_NOISE, len(timesteps)].
    """
    B = z0_batch.shape[0]
    z0 = z0_batch.to(device, dtype=dtype)
    prompt_emb = prompt_emb_batch.to(device, dtype=dtype)
    pooled_emb = pooled_emb_batch.to(device, dtype=dtype)

    time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]], dtype=dtype, device=device
    ).expand(B, -1)

    added_cond_kwargs = {
        "text_embeds": pooled_emb,
        "time_ids": time_ids,
    }

    mse_per_noise_t = torch.zeros(B, len(fixed_noises), len(timesteps))

    for ti, t in enumerate(timesteps):
        alpha_t = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_t).sqrt()

        for ni in range(len(fixed_noises)):
            # noise = fixed_noises[ni].to(device, dtype=dtype).expand(B, -1, -1, -1)
            noise = torch.randn(B, 4, z0.shape[2], z0.shape[3], device=device, dtype=dtype)







            z_t = sqrt_alpha * z0 + sqrt_one_minus_alpha * noise
            noise_pred = unet(
                z_t, t,
                encoder_hidden_states=prompt_emb,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            mse = (noise - noise_pred).pow(2).mean(dim=[1, 2, 3]).cpu()
            mse_per_noise_t[:, ni, ti] = mse

            del z_t, noise_pred, mse

    return mse_per_noise_t


def main():
    validate_runtime_config()
    from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    start_time = time.time()

    # Prepare the fixed-noise bank used by all candidate models in this family.
    print("=" * 70)
    print("  Improved MSE attribution: BLIP captions, multiple timesteps, aggregation, and fixed noise.")
    print("  Adapted for SDXL-family models; each candidate loads its own components.")
    print("  Use fp32 for the VAE to avoid NaNs, and fp16 for UNet/TextEncoder.")
    print("=" * 70)
    print(f"  USE_BLIP={USE_BLIP}, number of timesteps={len(TIMESTEPS)}")
    print(f"  noise samples={NUM_NOISE}, NOISE_SEED={NOISE_SEED}, batch size={BATCH_SIZE}")
    print(f"  image resolution={IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  UNet/TextEnc dtype={DTYPE}, VAE dtype={VAE_DTYPE}")

    # ====== Build the fixed-noise bank ======
    latent_h = IMAGE_SIZE // 8
    print(f"\n[0] Build the fixed-noise bank (seed={NOISE_SEED}, latent={latent_h}x{latent_h})...")
    rng = torch.Generator()
    rng.manual_seed(NOISE_SEED)
    fixed_noises = [
        torch.randn(1, 4, latent_h, latent_h, generator=rng)
        for _ in range(NUM_NOISE)
    ]
    print(f"  Generated {NUM_NOISE} fixed noise samples, shape={fixed_noises[0].shape}")

    # ====== Collect all image paths ======
    print(f"\n[1] Collect image paths...")
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

    # ====== Generating BLIP captions ======
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
    for t in TIMESTEPS:
        csv_cols.append(f'mse_t{t}')
    with open(RESULTS_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(csv_cols)

    # ====== Process each candidate model ======
    print(f"\n[3] Load each candidate component set, encode inputs, and compute MSE...")

    all_results = {}
    all_mse_profiles = {}

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n{'=' * 70}")
        print(f"  Processing model: {model_name} ({model_id})")
        print(f"{'=' * 70}")
        model_start = time.time()

        print(f"  [a] Loading VAE (fp32, avoid NaNs)...")
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder='vae',
            torch_dtype=VAE_DTYPE
        ).to(DEVICE)
        vae.eval()
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
        print(f"      [OK] VAE loaded (dtype={VAE_DTYPE}, scaling_factor={scaling_factor})")

        print(f"  [b] Encoding images into latent space (VAE fp32 -> latents fp16)...")
        model_latents = {}
        for source, paths in all_paths.items():
            print(f"      Encoding {source}...", end=' ', flush=True)
            latents = encode_images_to_latents(
                vae, paths, BATCH_SIZE, DEVICE, VAE_DTYPE, DTYPE, scaling_factor
            )
            model_latents[source] = latents
            print(f"shape={latents.shape}, dtype={latents.dtype}")

            nan_count = torch.isnan(latents).sum().item()
            if nan_count > 0:
                print(f"      [WARN] WARNING: {source} has {nan_count} NaN values!")
            else:
                print(f"      [OK] {source} contains no NaNs")

        del vae
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  [c] Loading dual text encoders (fp16)...")

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer_2')

        text_encoder = load_fp16_or_fallback(
            CLIPTextModel, model_id, 'text_encoder', DEVICE, DTYPE
        )
        text_encoder.eval()

        text_encoder_2 = load_fp16_or_fallback(
            CLIPTextModelWithProjection, model_id, 'text_encoder_2', DEVICE, DTYPE
        )
        text_encoder_2.eval()

        print(f"  [d] Encoding text embeddings...")
        model_prompt_embs = {}
        model_pooled_embs = {}
        for source, paths in all_paths.items():
            prompts = []
            for p in paths:
                if USE_BLIP and p in captions:
                    prompts.append(captions[p])
                else:
                    prompts.append("")
            prompt_embs, pooled_embs = encode_prompts_sdxl(
                tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                prompts, DEVICE, DTYPE
            )
            model_prompt_embs[source] = prompt_embs
            model_pooled_embs[source] = pooled_embs
            print(f"      {source}: prompt_embs={prompt_embs.shape}, pooled={pooled_embs.shape}")

        del text_encoder, text_encoder_2, tokenizer, tokenizer_2
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  [e] Loading UNet (fp16) and scheduler...")
        unet = load_fp16_or_fallback(
            UNet2DConditionModel, model_id, 'unet', DEVICE, DTYPE
        )
        unet.eval()
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')

        print(f"  [f] Computing multi-timestep MSE...")
        for source in SOURCE_FOLDERS:
            if source not in model_latents:
                continue
            latents = model_latents[source]
            prompt_embs = model_prompt_embs[source]
            pooled_embs = model_pooled_embs[source]
            n = latents.shape[0]

            pbar = tqdm(
                range(0, n, BATCH_SIZE),
                desc=f'      {source}->{model_name}', ncols=80
            )
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, n)
                batch_z0 = latents[batch_start:batch_end]
                batch_prompt = prompt_embs[batch_start:batch_end]
                batch_pooled = pooled_embs[batch_start:batch_end]

                mse_per_noise_t = compute_mse_multi_timestep(
                    batch_z0, batch_prompt, batch_pooled,
                    unet, scheduler, TIMESTEPS, fixed_noises,
                    DEVICE, DTYPE
                )

                rows = []
                for j in range(mse_per_noise_t.shape[0]):
                    idx = batch_start + j
                    noise_arr = mse_per_noise_t[j].numpy()  # [NUM_NOISE, len(TIMESTEPS)]
                    mse_arr = noise_arr.mean(axis=0)         # [len(TIMESTEPS)]
                    avg_mse = float(mse_arr.mean())

                    all_results[(source, idx, model_name)] = avg_mse
                    all_mse_profiles[(source, idx, model_name)] = mse_arr

                    for ni in range(noise_arr.shape[0]):
                        row = [source, idx, model_name, ni]
                        for ti in range(len(TIMESTEPS)):
                            row.append(f"{noise_arr[ni, ti]:.10f}")
                        rows.append(row)

                with open(RESULTS_CSV, 'a', newline='') as f:
                    csv.writer(f).writerows(rows)

                del batch_z0, batch_prompt, batch_pooled, mse_per_noise_t

        del unet, scheduler, model_latents, model_prompt_embs, model_pooled_embs
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - model_start
        print(f"  model {model_name} completed, elapsed {elapsed:.1f}s")

    # ====== Compute accuracy ======
    print(f"\n[4] Computing attribution accuracy...")

    num_per_source = {folder: len(paths) for folder, paths in all_paths.items()}

    correct_avg = 0
    total = 0
    confusion_avg = {s: {m: 0 for m in SOURCE_FOLDERS} for s in SOURCE_FOLDERS}

    for source in SOURCE_FOLDERS:
        if source not in num_per_source:
            continue
        n = num_per_source[source]
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

    per_t_correct = {t: 0 for t in TIMESTEPS}
    per_t_total = {t: 0 for t in TIMESTEPS}

    for source in SOURCE_FOLDERS:
        if source not in num_per_source:
            continue
        n = num_per_source[source]
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

    best_t = max(TIMESTEPS, key=lambda t: per_t_correct[t] / per_t_total[t] if per_t_total[t] > 0 else 0)
    acc_best_t = per_t_correct[best_t] / per_t_total[best_t] if per_t_total[best_t] > 0 else 0

    weight_schemes = {
        'uniform':     np.ones(len(TIMESTEPS)),
        'low_t_heavy': np.array([1.0 / (t + 1) for t in TIMESTEPS]),
        'low_only':    np.array([1.0 if t <= 200 else 0.0 for t in TIMESTEPS]),
        'mid_only':    np.array([1.0 if 100 <= t <= 500 else 0.0 for t in TIMESTEPS]),
        'high_only':   np.array([1.0 if t >= 500 else 0.0 for t in TIMESTEPS]),
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
            if source not in num_per_source:
                continue
            n = num_per_source[source]
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

    print(f"\n{'=' * 70}")
    print(f"  Summary (USE_BLIP={USE_BLIP}, NOISE_SEED={NOISE_SEED})")
    print(f"{'=' * 70}")

    print(f"\n  Method A - average MSE over all timesteps: {acc_avg:.4f} ({correct_avg}/{total})")

    print(f"\n  Method B - per-timestep accuracy:")
    for t in TIMESTEPS:
        if per_t_total[t] > 0:
            acc_t = per_t_correct[t] / per_t_total[t]
            marker = " <- BEST" if t == best_t else ""
            print(f"    t={t:4d}: {acc_t:.4f}{marker}")

    print(f"\n  Method C - best single timestep: t={best_t}, acc={acc_best_t:.4f}")

    print(f"\n  Method D - weighted schemes:")
    for scheme_name, acc_s in scheme_results.items():
        marker = " <- BEST" if scheme_name == best_scheme_name else ""
        print(f"    {scheme_name:>15}: {acc_s:.4f}{marker}")

    print(f"\n  Confusion matrix (average MSE over all timesteps):")
    for s in SOURCE_FOLDERS:
        row = f"{s:>15}" + "".join(f"{confusion_avg[s][m]:>15}" for m in SOURCE_FOLDERS)
        print(f"  {row}")

    summary = {
        'use_blip': USE_BLIP,
        'noise_seed': NOISE_SEED,
        'accuracy_avg_mse': acc_avg,
        'accuracy_best_single_t': {'t': best_t, 'acc': acc_best_t},
        'accuracy_per_timestep': {
            str(t): per_t_correct[t] / per_t_total[t] if per_t_total[t] > 0 else 0
            for t in TIMESTEPS
        },
        'accuracy_weighted_schemes': scheme_results,
        'best_weighted_scheme': {'name': best_scheme_name, 'acc': best_scheme_acc},
        'confusion_matrix_avg': confusion_avg,
        'total': total,
        'config': {
            'timesteps': TIMESTEPS,
            'num_noise': NUM_NOISE,
            'noise_seed': NOISE_SEED,
            'num_images': NUM_IMAGES,
            'batch_size': BATCH_SIZE,
            'image_size': IMAGE_SIZE,
            'vae_dtype': str(VAE_DTYPE),
            'unet_dtype': str(DTYPE),
        },
        'total_time': time.time() - start_time,
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total elapsed: {(time.time() - start_time) / 60:.1f} min")
    print(f"  Outputs: {RESULTS_CSV}, {SUMMARY_JSON}")


if __name__ == '__main__':
    main()
