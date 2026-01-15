# wearable-shoulder-kinematics

This repository accompanies the manuscript  
**“Fusing IMUs and Soft Sensors for Long-Duration Tracking of 3D Shoulder Kinematics with Minimal Calibration”**  
and provides code and data to support computational reproducibility and figure regeneration.

---

## Reproducibility and Figure Generation

Two notebooks are provided:

- **`notebooks/01_reproducible_run.ipynb`** implements the full sensor fusion processing pipeline (FIS), including data preprocessing, CNN training using a short calibration segment, application of the fusion algorithm to correct IMU drift, and evaluation of orientation errors for IMU-only, SS-only, and FIS estimates.

- **`notebooks/02_generate_paper_figures.ipynb`** generates the main manuscript figures directly from `data/figure_data.xlsx`, allowing deterministic reproduction of reported results without re-running model training.

---

## Data Description

- **`participant_*.csv`**: Datasets from all participants containing time, soft sensor signals, IMU quaternions, optical motion capture (MCL) quaternions, and MCL synchronization pulses.  

- **`figure_data.xlsx`**: Preprocessed and aggregated data used to generate the figures reported in the manuscript.

---

## Source Code

- **`src/utils.py`**: Data loading and preprocessing utilities.

- **`src/fis.py`**: Definition of the lightweight CNN used to map soft sensor signals to orientation estimates, and implementation of the FIS for yaw drift correction and smoothing.

- **`src/metrics.py`**: Error computation and evaluation utilities.

---

## Environment Setup

An example Conda environment is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate fis
jupyter notebook
