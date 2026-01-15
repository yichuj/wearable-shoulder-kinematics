# data

## Data (not stored in this repository)

Due to GitHub file size limits, data files are not included in this repository.  
Please obtain the data from the external link provided in the repository root `README.md`.

### Expected files

- `participant_1.csv`
- `participant_2.csv`
- …
- `participant_6.csv`
- `figure_data.xlsx`

Place the data files in this folder before running the notebooks.

---

## participant_x.csv

Each `participant_x.csv` file contains a time-synchronized recording from a single participant, including:

- **time**: timestamp in seconds  
- **SS**: soft sensor signals recorded from the sensing shirt. These values are the digitized outputs of the capacitance sensing chip and are unitless  
- **IMU orientations**: orientation estimates from inertial measurement units, represented as unit quaternions in the order **(w, x, y, z)**  
- **MCL orientations**: optical motion capture (reference) orientations, also represented as unit quaternions in the order **(w, x, y, z)**  
- **MCL sync**: a binary (0/1) synchronization pulse used to segment calibration and evaluation trials  

For both IMU and MCL orientations, measurements are provided for the shoulder, arm, and torso. For example, IMU orientation columns follow the pattern:

imu_w, imu_x, imu_y, imu_z,
imu_arm_w, imu_arm_x, imu_arm_y, imu_arm_z,
imu_tor_w, imu_tor_x, imu_tor_y, imu_tor_z

The same naming convention is used for the corresponding MCL orientation columns.

---

## figure_data.xlsx

`figure_data.xlsx` contains the preprocessed data used to generate all figures in the main text and supplementary material of the manuscript.

Each worksheet corresponds to a specific figure or sub-figure and contains the values plotted in the paper, allowing deterministic reproduction of all reported figures without re-running model training.
