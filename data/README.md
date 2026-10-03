# Data Directory

This directory is a placeholder for the PTB-XL dataset.

## Dataset: PTB-XL v1.0.3

**Source:** https://physionet.org/content/ptb-xl/1.0.3/

PTB-XL is a large publicly available 12-lead ECG dataset.
It is distributed under the Open Database License (ODbL) and does not require registration to download.

## Required Directory Structure

After downloading and extracting PTB-XL, place the files as follows:

```
data/
+-- ptbxl/
    +-- ptbxl_database.csv       # 21,799 records with metadata and diagnostic codes
    +-- scp_statements.csv       # SCP code descriptions and superclass mappings
    +-- records100/              # 100 Hz WFDB waveform files (.dat + .hea pairs)
        +-- 00000/
        +-- 01000/
        +-- 02000/
        +-- ...
```

## Important Notes

- Use the 100 Hz waveforms in `records100/`. Do NOT use `records500/`.
- Do not rename or restructure the files. All code resolves paths via `configs/config.py`.
- The full dataset is approximately 1.7 GB (100 Hz version).

## Citation

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022).
PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3).
PhysioNet. https://doi.org/10.13026/kfzx-aw45
