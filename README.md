# Introduction

This repository hosts the official PyTorch implementation of the paper:

**“DNA: Tracing Generated Images to Source Models via Dual-stage Native Attribution”**

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
...
```

Select the script that matches the model family predicted by AEDR.

| Script | Candidate source models included in the script |
| --- | --- |
| `Family_SD1.py` | `SD1.1`, `SD1.2`, `SD1.3`, `SD1.4`, `SD1.5` |
| `Family_SD2.py` | `SD2-base`, `SD2.1-base`, `SD2-typography`, `SD2-cartoon` |
| `Family_SD3.py` | `SD3-M`, `SD3.5-M`, `SD3.5-L`, `SD3.5-LT` |
| `Family_SDXL.py` | `SDXL-0.9`, `SDXL-1.0`, `SSD-1B`, `Segmind-Vega` |
| `Family_FLUX1.py` | `FLUX.1-dev`, `FLUX.1-Krea`, `FLUX.1-Lite`, `Chroma1-HD` |
| `Family_FLUX2.py` | `FLUX.2-dev`, `FLUX.2-klein-base-9B`, `FLUX.2-klein-base-4B` |

Before execution, edit the script and set the required runtime parameters. The public release intentionally leaves experiment-specific values empty or generic, so users should fill them according to their local dataset and evaluation setting.

| Parameter | Meaning | Reference value |
| --- | ---- | -- |
| `IMAGE_ROOT` | Root directory of the images to be attributed. | `"/path/to/your/images"` |
| `NUM_IMAGES` | Maximum number of images loaded from each source folder. | `500` |
| `NUM_NOISE` | Number of noise samples used for each image-model pair. | `5`, `10`, `15`, `...` |
| `TIMESTEPS` | Selected diffusion timesteps. | `[1, 11, 21, 31, 41, ...]` |
| `SIGMAS` | Selected flow-matching noise levels. | `[0.0621, 0.0825, 0.1028, 0.1232, 0.1436, ...]` |
| `NOISE_SEED` | Random seed for reproducible fixed-noise generation. | `123` |

Each script writes a CSV file that is used by the final accuracy evaluation:

```text
DNA_SD1_results.csv
DNA_SD2_results.csv
...
```

## Step 3: Evaluate Attribution Accuracy

Use `Evaluate_Accuracy.py` in test mode to compute attribution accuracy from the CSV files produced by the DNA family scripts.

Before running test mode, fill in each entry with the finalized parameters for that family:

| Parameter | Meaning | Reference value |
| --- | --- | --- |
| `csv_path` | CSV file produced by the corresponding DNA family script. | `"DNA_SD1_results.csv"` |
| `n_noise` | Number of noise samples to read per image-model pair from the CSV. | `5`, `10`, `15`, `...` |
| `timesteps` | Selected discriminative timestep or sigma-index columns. | `[1, 11, 21, 31, 41, ...]` |
| `bias` | Model-wise bias correction selected from validation experiments. | `[0.0, -0.5, -0.3, ...]` |

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
    ...
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
  Bias: [0.0, -0.5, -0.3, ...]

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
