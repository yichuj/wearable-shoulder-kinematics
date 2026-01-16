from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


def get_OMC_idx(sync):
    """
    Extract trial index ranges from a binary sync channel.

    The sync signal is expected to be 0/1, where rising edges mark trial start
    and falling edges mark trial end. Returned indices are half-open intervals
    [start, end), suitable for NumPy slicing.

    Parameters
    ----------
    sync : array-like, shape (N,)
        Binary sync channel.

    Returns
    -------
    list of tuple
        List of (start_idx, end_idx) pairs, where end_idx is exclusive.
    """
    diff = np.diff(sync)
    srt_idx, end_idx = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0]
    if sync[0] == 1:
        srt_idx = np.insert(srt_idx, 0, 0)
    if sync[-1] == 1:
        end_idx = np.append(end_idx, len(sync) - 1)
    return list(zip(srt_idx, end_idx + 1))


def prc_quat(quat):
    """
    Canonicalize unit quaternions by enforcing a consistent hemisphere.

    This function flips the sign of quaternions whose scalar component (w)
    is negative, ensuring w >= 0. This removes the q vs. -q sign ambiguity
    while preserving the represented rotation.

    Parameters
    ----------
    quat : array-like, shape (N, 4)
        Unit quaternions in (w, x, y, z) order.

    Returns
    -------
    np.ndarray, shape (N, 4)
        Quaternions with w >= 0.
    """
    quat = np.array(quat)
    quat[quat[:, 0] < 0, :] = -quat[quat[:, 0] < 0, :]
    return quat


def read_csv(filename):
    """
    Load a participant CSV file and parse channels into arrays.

    Assumes the CSV contains a header row and columns arranged as:
      0: time
      1-8: soft sensor (SS) channels
      9-12: IMU quaternion
      13-16: IMU_arm quaternion
      17-20: IMU_tor quaternion
      21-24: OMC quaternion
      25-28: OMC_arm quaternion
      29-32: OMC_tor quaternion
      last column: OMC sync channel (0/1)

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the CSV file.

    Returns
    -------
    tuple
        (t, SS, IMU, IMU_arm, IMU_tor, OMC, OMC_arm, OMC_tor, OMC_idx), where:
        - t: np.ndarray, shape (N,)
        - SS: np.ndarray, shape (N, 8)
        - IMU, IMU_arm, IMU_tor, OMC, OMC_arm, OMC_tor: np.ndarray, shape (N, 4)
        - OMC_idx: list of (start, end) index pairs (end exclusive)
    """
    data = pd.read_csv(filename, header=0).to_numpy()
    time = np.array(data[:, 0])
    SS = np.array(data[:, 1:9])
    OMC_idx = get_OMC_idx(data[:, -1])

    IMU = prc_quat(data[:, 9:13])
    IMU_arm = prc_quat(data[:, 13:17])
    IMU_tor = prc_quat(data[:, 17:21])

    OMC = prc_quat(data[:, 21:25])
    OMC_arm = prc_quat(data[:, 25:29])
    OMC_tor = prc_quat(data[:, 29:33])

    return time, SS, IMU, IMU_arm, IMU_tor, OMC, OMC_arm, OMC_tor, OMC_idx


def prc_data(data):
    """
    Preprocess raw signals for model input.

    Steps:
    1) Fit MinMax normalization on SS using the first detected trial segment,
       then apply the transform to the full SS sequence.
    2) Reshape SS to (N, C, 1) using a length-1 sliding window, matching the
       expected input format for downstream models.
    3) Clamp trial indices to valid bounds.

    Parameters
    ----------
    data : tuple
        Output tuple from `read_csv`.

    Returns
    -------
    tuple
        Preprocessed data tuple with the same structure as the input, except:
        - SS is normalized and reshaped to (N, 8, 1)
        - OMC_idx is clamped to [0, len(t)]
    """
    t, SS, IMU, IMU_arm, IMU_tor, OMC, OMC_arm, OMC_tor, OMC_idx = data

    # Normalize SS using the first trial segment as calibration.
    scaler = MinMaxScaler().fit(SS[OMC_idx[0][0] : OMC_idx[0][1], :])
    SS = scaler.transform(SS)

    # Reshape SS to (N, C, 1).
    SS = np.transpose(sliding_window_view(SS, 1, 0), (0, 2, 1))

    # Clamp OMC sync ranges to valid array bounds.
    OMC_idx = [(max(0, x), min(max(0, y), len(t))) for (x, y) in OMC_idx]

    return t, SS, IMU, IMU_arm, IMU_tor, OMC, OMC_arm, OMC_tor, OMC_idx
