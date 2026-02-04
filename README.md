# T2-Weighted Imaging of Water, Fat and Silicone

## Overview

This repository implements the **joint multi-echo reconstruction and water–fat–silicone (WFS) separation** framework described in the accompanying paper. The code reconstructs water, fat, and silicone images directly from undersampled multi-echo FSE k-space data by incorporating chemical shift modelling in the forward model.

Compared to conventional two-stage pipelines (reconstruct echoes → separate species), the proposed approach exploits correlations across echoes and enforces species-specific sparsity, enabling high acceleration factors.

## The method

The main pipeline (`scripts/main_wfs.py`) follows the paper’s Methods:

1. **Multi-echo data handling**  
   Loads multi-echo FSE k-space data and groups positive/negative polarities for joint processing.

2. **Calibration and field mapping**  
   Uses ACS data to estimate ESPIRiT coil sensitivities and a low-resolution field map, which are fixed during reconstruction.

3. **Joint reconstruction**  
   Reconstructs water, fat, and silicone images from undersampled k-space, incorporating coil sensitivities, echo times, fieldmap estimate, chemical-shift encoding, and bipolar chemical shift-related displacement correction into a single forward model with species-specific sparsity regularization.

4. **Baselines**  
   Supports a conventional two-stage reconstruction for comparison with the joint method.

## Citation

If you use this code, please cite:

> Nurdinova A, Zhou X, Oscanoa JA, Shah P, Setsompop K, Daniel BL, Hargreaves BA.  
> **T2-Weighted Imaging of Water, Fat and Silicone.**  
> *Magnetic Resonance in Medicine*, 2025.


## Repository Layout

```
water_fat_silicone/
├── pyproject.toml
├── scripts/
│   └── main_wfs.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── import_data.py
│   ├── wfs_recon_fun.py
│   └── utils.py
└── README.md
```


## Install

You can use the environment attached:

```
micromamba create -f environment.yml
micromamba activate wfs_recon
```

From the repository root:

```
pip install -e .
```

This installs the `src` package in editable mode, so code changes take effect immediately.

---

## Running

Run scripts from CLI OR as a notebook by converting with jupytext:

```
python scripts/main_wfs.py
```

## Maintainer

Aizada Nurdinova (nurdaiza@stanford.edu)
