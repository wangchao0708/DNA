# Example Data

`DNA_SD1_results.csv` is a compact SD1-family attribution-output example.
`expected_sd1_output.txt` records the expected evaluation summary for this file.

It contains:

- 5 source-model classes: `SD1.1`, `SD1.2`, `SD1.3`, `SD1.4`, `SD1.5`
- 10 images per source class
- 30 noise samples per image-model pair
- 25 selected timesteps

The first entry in `Evaluate_Accuracy.py::TEST_CONFIGS` is configured to run on this file.
