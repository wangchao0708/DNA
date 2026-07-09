# DNA

This repository hosts the official PyTorch implementation of the paper:

**DNA: Tracing Generated Images to Source Models via Dual-stage Native Attribution**

The released files contain the core attribution scripts used by DNA. Before running the code, users should configure their own image paths, selected timesteps or sigmas, noise counts, random seeds, and output settings.

## Environment

Create and activate the conda environment:

```bash
conda create -n DNA python=3.12.0
conda activate DNA
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Step 1: Identify the Model Family with AEDR

Run AEDR first to determine the target model family. Please refer to [wangchao0708/AEDR](https://github.com/wangchao0708/AEDR) for the AEDR workflow.

## Step 2: Run the DNA Attribution Script for the Identified Family

After AEDR identifies the model family, run the corresponding DNA script. For example:

```bash
python Family_SD1.py
python Family_SD2.py
# ...
```

Select the script that matches the model family predicted by AEDR.

| Script | Candidate source models included in the script |
| --- | --- |
| `Family_SD1.py` | `CompVis/stable-diffusion-v1-1`, `CompVis/stable-diffusion-v1-2`, `CompVis/stable-diffusion-v1-3`, `CompVis/stable-diffusion-v1-4`, `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| `Family_SD2.py` | `sd2-community/stable-diffusion-2-base`, `sd2-community/stable-diffusion-2-1-base`, `pbevan11/stable-diffusion-2-typography`, `Norod78/sd2-cartoon-blip` |
| `Family_SD3.py` | `stabilityai/stable-diffusion-3-medium-diffusers`, `stabilityai/stable-diffusion-3.5-large`, `stabilityai/stable-diffusion-3.5-large-turbo`, `stabilityai/stable-diffusion-3.5-medium` |
| `Family_SDXL.py` | `stabilityai/stable-diffusion-xl-base-1.0`, `stabilityai/stable-diffusion-xl-base-0.9`, `segmind/SSD-1B`, `segmind/Segmind-Vega` |
| `Family_FLUX1.py` | `lodestones/Chroma1-HD`, `Freepik/flux.1-lite-8B`, `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-Krea-dev` |
| `Family_FLUX2.py` | `black-forest-labs/FLUX.2-dev`, `black-forest-labs/FLUX.2-klein-base-9B`, `black-forest-labs/FLUX.2-klein-base-4B` |

Before execution, edit the script and set the required runtime parameters. The public release intentionally leaves experiment-specific values empty or generic, so users should fill them according to their local dataset and evaluation setting.

| Parameter to edit | Meaning | Reference value |
| --- | --- | --- |
| `IMAGE_ROOT` | Root directory of the images to be attributed. Each source model should have its own subfolder under this directory. | `"/path/to/your/images"` or `DNA_IMAGE_ROOT=/path/to/your/images` |
| `NUM_IMAGES` | Maximum number of images loaded from each source folder. `None` uses all available images. | `None` or `500` |
| `NUM_NOISE` | Number of noise samples used for each image-model pair. Larger values improve stability but increase computation. | `5`, `10`, or `30` |
| `TIMESTEPS` | Selected diffusion timesteps for SD1, SD2, and SDXL scripts. | `[1, 11, 21, 31, 41]` |
| `SIGMAS` | Selected flow-matching noise levels for SD3, FLUX.1, and FLUX.2 scripts. | `[0.0621, 0.0825, 0.1028, 0.1232, 0.1436]` |
| `NOISE_SEED` | Random seed for reproducible fixed-noise generation. | `42` |
| `BATCH_SIZE` | Batch size for UNet/Transformer inference. Adjust this value if GPU memory is limited. | `20`, `25`, or `100` |
| `VAE_BATCH_SIZE` | Batch size for VAE encoding in FLUX-family scripts. This usually needs to be small. | `1` |
| `USE_BLIP` | Whether to use BLIP-2 generated captions as prompts. If `False`, empty prompts are used. | `False` for unconditional evaluation, `True` when caption conditioning is needed |

Each script writes a CSV file that is used by the final accuracy evaluation:

```text
DNA_SD1_results.csv
DNA_SD2_results.csv
...
```

The remaining families follow the same naming style, for example `DNA_SD3_results.csv`, `DNA_SDXL_results.csv`, `DNA_FLUX1_results.csv`, and `DNA_FLUX2_results.csv`.

## Step 3: Evaluate Attribution Accuracy

Use `Evaluate_Accuracy.py` in test mode to compute attribution accuracy from the CSV files produced by the DNA family scripts.

The default `TEST_CONFIGS` entries already use the same CSV names listed above. Before running test mode, fill in each entry with the finalized parameters for that family:

| Parameter | Meaning | Reference value |
| --- | --- | --- |
| `csv_path` | CSV file produced by the corresponding DNA family script. The file name should match the output of Step 2. | `"DNA_SD1_results.csv"` |
| `n_noise` | Number of noise samples to read per image-model pair from the CSV. This should be the same as `NUM_NOISE` used in Step 2. | `5`, `10`, or `30` |
| `score_type` | Scoring rule used to aggregate the MSE profile. Supported values include `avg`, `zscore`, `median`, `rank`, and `weighted`. | `"zscore"` |
| `timesteps` | Selected timestep or sigma-index columns used for evaluation when `score_type` is not `weighted`. These values must exist in the CSV columns. | `[1, 11, 21, 31, 41]` |
| `low_steps` / `high_steps` | Step groups used only when `score_type="weighted"`. | `low_steps=[1, 11, 21]`, `high_steps=[101, 111]` |
| `weight_low` | Weight assigned to the low-step group for weighted scoring. | `0.5` |
| `bias` | Model-wise bias correction learned or selected from validation experiments. Its length must match the number of candidate models in the family. | `[0.0, -0.5, -0.3, ...]` |

Example test configuration:

```python
TEST_CONFIGS = [
    {
        "csv_path": "DNA_SD1_results.csv",
        "n_noise": 30,
        "score_type": "zscore",
        "timesteps": [1, 11, 21, 31, 41],
        "bias": [0.0, -0.5, -0.3, -0.2, -0.1],
    },
    {
        "csv_path": "DNA_SD2_results.csv",
        "n_noise": 5,
        "score_type": "zscore",
        "timesteps": [61, 71, 81, 101, 111],
        "bias": [0.0, -0.4, -0.2, -0.1],
    },
    # ...
]
```

Run:

```bash
python Evaluate_Accuracy.py
```

## Example Test Output

A typical test-mode output has the following structure:

```text
======================================================================
  Model attribution [test mode]
  GPU: Yes
  number of test configurations: 6
======================================================================

======================================================================
  [TEST] file: DNA_SD1_results
  noise count: 30  score type: zscore
======================================================================
  models(5): ['SD1.1', 'SD1.2', 'SD1.3', 'SD1.4', 'SD1.5']
  images: 2500  timesteps: 25  range: [1, 241]
  class distribution: {'SD1.1': 500, 'SD1.2': 500, 'SD1.3': 500, 'SD1.4': 500, 'SD1.5': 500}
  timesteps (25 steps): [1, 11, 21, 31, 41, ...]
  Bias: [0.000000, ...]

  -- Results --
  Accuracy (no bias): 0.9120  (2280/2500)
  Accuracy (with bias): 0.9480  (2370/2500)

  -- Per-class accuracy (with bias) --
    SD1.1: 0.9500  (475/500)
    SD1.2: 0.9440  (472/500)
    SD1.3: 0.9520  (476/500)
    SD1.4: 0.9460  (473/500)
    SD1.5: 0.9480  (474/500)

  -- Confusion matrix (with bias, rows=true, columns=predicted) --
```

The exact numbers depend on the selected family, dataset, noise count, timesteps or sigmas, and bias values.
