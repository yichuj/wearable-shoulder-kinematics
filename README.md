# wearable-shoulder-kinematics
This repository accompanies the manuscript  
**“Fusing IMUs and Soft Sensors for Long-Duration Tracking of 3D Shoulder Kinematics with Minimal Calibration”**  
and provides code and data to support computational reproducibility and figure regeneration.

---

## Repository Contents

data/
  participant_*.csv Example participant datasets used for reproducible runs
  figure_data.xlsx Aggregated data used to generate paper figures

notebooks/
  01_reproducible_run.ipynb
  02_generate_paper_figures.ipynb

src/
  utils.py Data loading and preprocessing
  fis.py CNN model and fusion algorithm (FIS)
  metrics.py Error computation and evaluation

environment.yml Example conda environment

---

## Reproducible Run

The notebook  
**`notebooks/01_reproducible_run.ipynb`**  
implements the full pipeline on an example participant dataset:

1. Load and preprocess soft sensor, IMU, and motion capture data  
2. Train a lightweight CNN using a short calibration segment  
3. Apply the fusion algorithm to correct IMU drift  
4. Compute and visualize orientation errors for:
   - IMU-only
   - Soft-sensor-only
   - Fused (IMU + soft sensors)

---

## Figure Generation

The notebook  
**`notebooks/02_generate_paper_figures.ipynb`**  
generates the main manuscript figures directly from  
`data/figure_data.xlsx`.

---

## Data Description

- **`participant_*.csv`**  
  Datasets from all participants containing time, soft sensor signals, IMU quaternions, optical motion capture (MCL) quaternions, and MCL synchronization pulses.  

- **`figure_data.xlsx`**  
  Preprocessed and aggregated data used to generate the figures reported in the manuscript.

---

## Environment Setup

An example Conda environment is provided in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate fis
jupyter notebook
