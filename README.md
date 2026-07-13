# Introduction

This repository hosts the official PyTorch implementation of the paper:

**“DNA: Dual-stage Native Attribution for Generated Image Source Tracing”**

The released files contain the core attribution scripts used by DNA. Before running the code, users should configure their own image paths, selected timesteps or sigmas, noise counts, random seeds, and output settings.

## Paper-to-Code Mapping

| Paper | Code |
| --- | --- |
| AEDR | Please refer to [wangchao0708/AEDR](https://github.com/wangchao0708/AEDR) |
| NPC | `Family_SD1.py`, `Family_SD2.py`, `Family_SD3.py`, `Family_SDXL.py`, `Family_FLUX1.py`, `Family_FLUX2.py` |
| Output | `RESULTS_CSV` written by each `Family_*.py` script |
| Evaluation | `Evaluate_Accuracy.py` |

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

The family scripts run in Hugging Face offline mode by default:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
```

This assumes that all candidate models have already been downloaded to the local Hugging Face cache. If a model is not cached, disable offline mode or download the required checkpoints before running the script. Some model repositories may also require Hugging Face access approval.

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

Each family script expects one folder per source model under `IMAGE_ROOT`. For example, an SD1-style image directory can be organized as:

```text
/path/to/your/images/
  SD1.1/
    000001.png
    000002.png
    ...
  SD1.2/
    000001.png
    000002.png
    ...
  SD1.3/
    000001.png
    000002.png
    ...
  ...
```

Before execution, edit the script and set the required runtime parameters. The public release leaves experiment-specific values generic, so users should fill them according to their local dataset and evaluation setting.

| Parameter | Meaning | Reference value |
| --- | --- | --- |
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

Use `Evaluate_Accuracy.py` to compute attribution accuracy from the CSV files produced by the DNA family scripts. The script keeps only the paper setting: z-score normalized D-style scoring over the selected timesteps or sigmas, followed by model-wise bias correction.

Before running evaluation, fill in each entry with the finalized parameters for that family:

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
        "csv_path": "examples/DNA_SD1_results.csv",
        "n_noise": 30,
        "timesteps": [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241],
        "bias": [0.0, -1.238396, -1.182738, -1.161866, -1.113165],
    },
    {
        "csv_path": "DNA_SD2_results.csv",
        "n_noise": 5,
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

## Configuration Templates

The `configs/` directory provides lightweight reference templates:

```text
configs/family_sd1_example.json
configs/evaluate_sd1_example.json
```

The scripts currently use Python constants for configuration. These JSON files are intended as readable templates; copy the relevant values into `Family_*.py` or `Evaluate_Accuracy.py` before running a full experiment.

## Minimal Runnable Example

The repository includes a compact SD1 example:

```text
examples/DNA_SD1_results.csv
```

This file contains 10 images per SD1 source model, 30 noise samples per image-model pair, and the 25 selected timesteps used in the example evaluation. The first entry in `TEST_CONFIGS` in `Evaluate_Accuracy.py` is already configured for this file.

Run:

```bash
python Evaluate_Accuracy.py
```

Expected result for the included example:

```text
File: DNA_SD1_results.csv
Models: ['SD1.1', 'SD1.2', 'SD1.3', 'SD1.4', 'SD1.5']
Images: 50
Selected steps: [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241]
------------------------------------------------------------------------
Accuracy without bias: 0.9400
Accuracy with bias:    0.9800

Per-class accuracy with bias:
  SD1.1: 1.0000 (10/10)
  SD1.2: 1.0000 (10/10)
  SD1.3: 1.0000 (10/10)
  SD1.4: 0.9000 (9/10)
  SD1.5: 1.0000 (10/10)
```

The full expected console summary is provided in:

```text
examples/expected_sd1_output.txt
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
