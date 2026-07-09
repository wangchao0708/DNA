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

# ========================== Configuration ==========================
IMAGE_ROOT = os.environ.get('DNA_IMAGE_ROOT', '')
SOURCE_FOLDERS = ['FLUX.2-dev', 'FLUX.2-klein-9B', 'FLUX.2-klein-4B']
MODEL_IDS = {
    'FLUX.2-klein-9B': 'black-forest-labs/FLUX.2-klein-base-9B',
    'FLUX.2-klein-4B': 'black-forest-labs/FLUX.2-klein-base-4B',
    'FLUX.2-dev':      'black-forest-labs/FLUX.2-dev',
}

NUM_IMAGES = None
NUM_NOISE = None
SIGMAS = []
NOISE_SEED = None
BATCH_SIZE = 20
VAE_BATCH_SIZE = 1
IMAGE_SIZE = 1024

LATENT_CHANNELS = 32
LATENT_H = IMAGE_SIZE // 8
LATENT_W = IMAGE_SIZE // 8
PATCHIFIED_CHANNELS = 128
PATCHIFIED_H = LATENT_H // 2
PATCHIFIED_W = LATENT_W // 2
PACKED_SEQ_LEN = PATCHIFIED_H * PATCHIFIED_W
PACKED_DIM = PATCHIFIED_CHANNELS

DTYPE = torch.bfloat16
USE_BLIP = True
RESULTS_CSV = 'DNA_FLUX2_results.csv'
SUMMARY_JSON = 'DNA_FLUX2_summary.json'
BLIP_CACHE = 'DNA_FLUX2_blip_captions.json'
MAX_SEQ_LEN = 512
GUIDANCE_SCALE = 4.0

VAE_DEVICE = 'cuda:0'
TEXT_DEVICE = 'cuda:0'
BLIP_DEVICE = 'cuda:0'


# ========================== Utility functions ==========================


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
    ])
    return transform(img)


def get_module_device(module):
    for p in module.parameters():
        return p.device
    return torch.device('cuda:0')


def get_transformer_first_device(transformer):
    try:
        for p in transformer.parameters():
            return p.device
    except Exception:
        pass
    return torch.device('cuda:0')


def generate_captions(image_paths, device='cuda:0'):
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


# ========================== Encoding functions ==========================

@torch.no_grad()
def encode_images_to_latents(pipe, image_paths, batch_size, device, dtype):
    all_latents = []
    img_ids_out = None

    pipe.vae.to(device)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = torch.stack([load_image_tensor(p, IMAGE_SIZE) for p in batch_paths])
        batch = batch.to(device, dtype=dtype)

        z = pipe.vae.encode(batch).latent_dist.mode()
        zp = pipe._patchify_latents(z)

        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(zp.device, zp.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1).to(zp.device, zp.dtype)
            + pipe.vae.config.batch_norm_eps
        )
        zp = (zp - bn_mean) / bn_std

        if img_ids_out is None:
            img_ids_out = pipe._prepare_latent_ids(zp[:1]).cpu()

        z_tokens = zp.flatten(2).transpose(1, 2).contiguous()
        all_latents.append(z_tokens.cpu())

        del batch, z, zp, z_tokens

    return torch.cat(all_latents, dim=0), img_ids_out


@torch.no_grad()
def encode_prompts_flux2(pipe, prompts, max_seq_len=512):
    text_device = get_module_device(pipe.text_encoder)

    all_prompt_embeds = []
    text_ids_out = None

    if pipe.__class__.__name__ == "Flux2Pipeline":
        text_layers = (10, 20, 30)
    else:
        text_layers = (9, 18, 27)

    for prompt in tqdm(prompts, desc="    encoding prompts", ncols=80):
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=torch.device(text_device),
            num_images_per_prompt=1,
            max_sequence_length=max_seq_len,
            text_encoder_out_layers=text_layers,
        )
        all_prompt_embeds.append(prompt_embeds.cpu())
        if text_ids_out is None:
            text_ids_out = text_ids.cpu()

    prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
    return prompt_embeds, text_ids_out


# ========================== MSE computation ==========================

@torch.no_grad()
def compute_mse_flux2(
    z0_batch,
    prompt_emb_batch,
    text_ids,
    img_ids,
    transformer,
    sigmas,
    fixed_noises,
    dtype,
    has_guidance=True,
    guidance_scale=4.0,
):
    """
    Returns: mse_per_noise_sigma [B, NUM_NOISE, len(sigmas)].
    """
    B = z0_batch.shape[0]
    input_device = get_transformer_first_device(transformer)

    z0 = z0_batch.to(input_device, dtype=dtype)
    prompt_emb = prompt_emb_batch.to(input_device, dtype=dtype)
    txt_ids_base = text_ids.to(input_device)
    img_ids_base = img_ids.to(input_device)

    mse_per_noise_sigma = torch.zeros(B, len(fixed_noises), len(sigmas), dtype=torch.float32)

    for si, sigma in enumerate(sigmas):
        for ni in range(len(fixed_noises)):
            noise = fixed_noises[ni].to(input_device, dtype=dtype).expand(B, -1, -1)

            z_t = (1.0 - sigma) * z0 + sigma * noise
            v_target = noise - z0
            timestep = torch.tensor([sigma] * B, device=input_device, dtype=dtype)

            kwargs = dict(
                hidden_states=z_t,
                timestep=timestep,
                encoder_hidden_states=prompt_emb,
                img_ids=img_ids_base.expand(B, -1, -1),
                txt_ids=txt_ids_base.expand(B, -1, -1),
                return_dict=False,
            )

            if has_guidance:
                kwargs['guidance'] = torch.tensor(
                    [guidance_scale] * B, device=input_device, dtype=dtype
                )

            v_pred = transformer(**kwargs)[0]
            mse = (v_target - v_pred).pow(2).mean(dim=[1, 2]).float().cpu()
            mse_per_noise_sigma[:, ni, si] = mse

            del noise, z_t, v_target, timestep, v_pred, mse

    return mse_per_noise_sigma


# ========================== Main entry point ==========================

def main():
    validate_runtime_config()
    from diffusers import Flux2Pipeline, Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    start_time = time.time()

    print("=" * 70)
    print("  Improved MSE attribution: flow matching, BLIP captions, multiple sigmas, and fixed noise.")
    print("  Adapted for FLUX.2 with multi-GPU loading (device_map='balanced' / 'auto').")
    print("=" * 70)
    print(f"  USE_BLIP={USE_BLIP}, number of sigmas={len(SIGMAS)}")
    print(f"  noise samples={NUM_NOISE}, NOISE_SEED={NOISE_SEED}")
    print(f"  batch size: Transformer={BATCH_SIZE}, VAE={VAE_BATCH_SIZE}")
    print(f"  latent_channels={LATENT_CHANNELS}, packed_dim={PACKED_DIM}, dtype={DTYPE}")
    print(f"  GUIDANCE_SCALE={GUIDANCE_SCALE}")
    print(f"  multi-GPU: VAE/Text->{VAE_DEVICE}, Transformer->device_map")

    # ====== [0/4] Build the fixed-noise bank ======
    print(f"\n[0/4] Build the fixed-noise bank (seed={NOISE_SEED})...")
    rng = torch.Generator(device='cpu')
    rng.manual_seed(NOISE_SEED)
    fixed_noises = []
    for _ in range(NUM_NOISE):
        noise = torch.randn(
            1, PACKED_SEQ_LEN, PACKED_DIM,
            generator=rng,
            dtype=DTYPE,
            device='cpu'
        )
        fixed_noises.append(noise)
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
        print("\n[1/4] Generating BLIP captions...")
        all_image_paths = []
        for paths in all_paths.values():
            all_image_paths.extend(paths)
        captions = generate_captions(all_image_paths, BLIP_DEVICE)
    else:
        print("\n[1/4] Skipping BLIP and using empty prompts")

    # ====== Prepare the CSV file ======
    csv_cols = ['source', 'img_idx', 'model', 'noise_idx']
    for s in SIGMAS:
        csv_cols.append(f'mse_t{int(s * 1000)}')
    with open(RESULTS_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(csv_cols)

    # ====== [2/4] Load each candidate pipeline, encode inputs, compute MSE, and release memory ======
    print(f"\n[2/4] Load each pipeline, encode inputs, and compute MSE...")

    all_results = {}
    all_mse_profiles = {}

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n  {'-' * 55}")
        print(f"  Loading pipeline: {model_name} ({model_id})")

        if model_name == "FLUX.2-dev":
            pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=DTYPE)
        else:
            pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=DTYPE)

        has_guidance = bool(getattr(pipe.transformer.config, 'guidance_embeds', False))
        print(f"  guidance_embeds: {has_guidance}"
              + (f"  (guidance_scale={GUIDANCE_SCALE})" if has_guidance else ""))

        print(f"  Encoding images into the official latent-token space (VAE -> {VAE_DEVICE})...")
        model_latents = {}
        model_img_ids = {}
        for folder, paths in all_paths.items():
            print(f"    Encoding {folder}...", end=' ', flush=True)
            latents, img_ids = encode_images_to_latents(
                pipe, paths, VAE_BATCH_SIZE, VAE_DEVICE, DTYPE
            )
            model_latents[folder] = latents
            model_img_ids[folder] = img_ids
            print(f"latents={latents.shape}, img_ids={img_ids.shape}")

        pipe.vae.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Moved VAE to CPU")

        print(f"  Encoding text embeddings (Text Encoder -> {TEXT_DEVICE})...")
        pipe.text_encoder.to(TEXT_DEVICE)
        model_prompt_embs = {}
        model_text_ids = {}
        for folder, paths in all_paths.items():
            prompts = [captions.get(p, "") if USE_BLIP else "" for p in paths]
            prompt_embs, text_ids = encode_prompts_flux2(pipe, prompts, MAX_SEQ_LEN)
            model_prompt_embs[folder] = prompt_embs
            model_text_ids[folder] = text_ids
            print(f"    {folder}: prompt_embs={prompt_embs.shape}, text_ids={text_ids.shape}")

        pipe.text_encoder.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Moved the text encoder to CPU and released GPU memory")

        print(f"  Loading Transformer (from_pretrained, device_map='balanced')...")

        _old_transformer = pipe.transformer
        pipe.transformer = None
        del _old_transformer
        gc.collect()

        transformer = Flux2Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=DTYPE,
            device_map="balanced",
        )
        transformer.eval()

        if hasattr(transformer, "hf_device_map"):
            _dmap = transformer.hf_device_map
            _devs = set(_dmap.values())
            print(f"  Transformer device map: {_devs}")

        model_start = time.time()

        for source in SOURCE_FOLDERS:
            if source not in model_latents:
                continue

            latents = model_latents[source]
            prompt_embs = model_prompt_embs[source]
            text_ids = model_text_ids[source]
            img_ids = model_img_ids[source]
            n = latents.shape[0]

            pbar = tqdm(range(0, n, BATCH_SIZE),
                        desc=f'    {source}->{model_name}', ncols=80)
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, n)
                batch_z0 = latents[batch_start:batch_end]
                batch_prompt = prompt_embs[batch_start:batch_end]

                mse_per_noise_sigma = compute_mse_flux2(
                    z0_batch=batch_z0,
                    prompt_emb_batch=batch_prompt,
                    text_ids=text_ids,
                    img_ids=img_ids,
                    transformer=transformer,
                    sigmas=SIGMAS,
                    fixed_noises=fixed_noises,
                    dtype=DTYPE,
                    has_guidance=has_guidance,
                    guidance_scale=GUIDANCE_SCALE,
                )

                rows = []
                for j in range(mse_per_noise_sigma.shape[0]):
                    idx = batch_start + j
                    noise_arr = mse_per_noise_sigma[j].numpy()  # [NUM_NOISE, len(SIGMAS)]
                    mse_arr = noise_arr.mean(axis=0)             # [len(SIGMAS)]
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

                del batch_z0, batch_prompt, mse_per_noise_sigma

        elapsed = time.time() - model_start
        print(f"  model {model_name} MSE computation completed, elapsed {elapsed:.1f}s")

        del transformer, pipe, model_latents, model_prompt_embs, model_text_ids, model_img_ids
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

    best_si = max(
        range(len(SIGMAS)),
        key=lambda i: per_s_correct[i] / per_s_total[i] if per_s_total[i] > 0 else 0
    )
    best_sigma = SIGMAS[best_si]
    acc_best_sigma = (per_s_correct[best_si] / per_s_total[best_si]
                      if per_s_total[best_si] > 0 else 0)

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
    header = ("{:>22}".format("true\\pred")
              + "".join(f"{m:>22}" for m in SOURCE_FOLDERS))
    print(f"  {header}")
    for s in SOURCE_FOLDERS:
        row = (f"{s:>22}"
               + "".join(f"{confusion_avg[s][m]:>22}" for m in SOURCE_FOLDERS))
        print(f"  {row}")

    summary = {
        'use_blip': USE_BLIP,
        'noise_seed': NOISE_SEED,
        'per_model_encoding': True,
        'dual_gpu': True,
        'accuracy_avg_mse': acc_avg,
        'accuracy_best_single_sigma': {
            'sigma': best_sigma, 'acc': acc_best_sigma
        },
        'accuracy_per_sigma': {
            f"{SIGMAS[i]:.4f}": (per_s_correct[i] / per_s_total[i]
                                 if per_s_total[i] > 0 else 0)
            for i in range(len(SIGMAS))
        },
        'accuracy_weighted_schemes': scheme_results,
        'best_weighted_scheme': {
            'name': best_scheme_name, 'acc': best_scheme_acc
        },
        'confusion_matrix_avg': confusion_avg,
        'total': total,
        'config': {
            'sigmas_count': len(SIGMAS),
            'sigma_range': [SIGMAS[0], SIGMAS[-1]],
            'num_noise': NUM_NOISE,
            'noise_seed': NOISE_SEED,
            'num_images': NUM_IMAGES,
            'batch_size': BATCH_SIZE,
            'vae_batch_size': VAE_BATCH_SIZE,
            'latent_channels': LATENT_CHANNELS,
            'packed_dim': PACKED_DIM,
            'packed_seq_len': PACKED_SEQ_LEN,
            'guidance_scale': GUIDANCE_SCALE,
            'max_seq_len': MAX_SEQ_LEN,
            'dtype': str(DTYPE),
            'vae_device': VAE_DEVICE,
            'text_device': TEXT_DEVICE,
            'transformer_device': 'device_map=balanced',
        },
        'total_time': time.time() - start_time,
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total elapsed: {(time.time() - start_time) / 60:.1f} min")
    print(f"  Outputs: {RESULTS_CSV}, {SUMMARY_JSON}")


if __name__ == '__main__':
    main()
