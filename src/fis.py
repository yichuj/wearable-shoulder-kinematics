import numpy as np
import quaternionic as qc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import (
    Input,
    Flatten,
    Dropout,
    Dense,
    Reshape,
    Conv2D,
    MaxPooling2D,
)


def set_tf_seed(seed):
    """
    Set random seeds and enable deterministic ops for Keras/TensorFlow reproducibility.

    Parameters
    ----------
    seed : int
        Seed used by Keras/TensorFlow random number generators.

    Notes
    -----
    Enabling deterministic ops can reduce training throughput on some systems.
    """
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def get_CNN(X, Y):
    """
    Construct a lightweight CNN used in the fusion model.

    The model expects input shaped like X with dimensions (N, T, C) and outputs
    a regression vector with dimensionality matching Y.shape[1].

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, T, C) used to infer model input shape.
    Y : np.ndarray
        Target array of shape (N, D) used to infer output dimensionality.

    Returns
    -------
    keras.Model
        Uncompiled Keras model (caller is expected to compile/train).
    """
    set_tf_seed(42)

    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(Reshape((X.shape[1], X.shape[2], 1)))
    model.add(Conv2D(64, (1, min(3, X.shape[2])), activation="relu", padding="same"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((1, min(2, X.shape[2]))))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dropout(0.25))
    model.add(Dense(Y.shape[1], activation="linear"))
    return model


def fusion_alg(quat_arm, quat_tor, quat_est, win=0):
    """
    Apply yaw-drift correction to estimate shoulder orientation (FIS).

    The function estimates IMU yaw drift using a drift-free shoulder
    orientation from soft sensors, extracts the yaw component of the drift,
    optionally smooths it using causal quaternion averaging, and applies the
    correction to the raw IMU orientations to obtain the final shoulder
    orientation.

    Parameters
    ----------
    quat_arm : array-like, shape (N, 4)
        Raw arm IMU orientation quaternions (w, x, y, z).
    quat_tor : array-like, shape (N, 4)
        Raw torso IMU orientation quaternions (w, x, y, z).
    quat_est : array-like, shape (N, 4)
        Drift-free shoulder orientation estimated from soft sensors.
    win : int, optional
        Trailing smoothing window length in samples. If win > 0, applies
        causal low-pass filtering to the yaw correction.

    Returns
    -------
    quat_out : quaternionic.array, shape (N,)
        Corrected shoulder orientation quaternion sequence.
    """

    # Convert from float array to quaternions array
    quat_arm = qc.array(np.ascontiguousarray(quat_arm, dtype=np.float64)).normalized
    quat_tor = qc.array(np.ascontiguousarray(quat_tor, dtype=np.float64)).normalized
    quat_est = qc.array(np.ascontiguousarray(quat_est, dtype=np.float64)).normalized

    # Find correction quaternions
    q = quat_tor * quat_est * np.conj(quat_arm)

    # Extract yaw
    yaw = np.arctan2(2*(q.x*q.y + q.w*q.z), 1 - 2*(q.y*q.y + q.z*q.z))

    # Convert back to quaternions (pure yaw)
    w, z = np.cos(yaw/2.0), np.sin(yaw/2.0)
    quat_corr = np.column_stack([w, np.zeros_like(w), np.zeros_like(w), z])
    quat_corr_sm = quat_corr.copy()

    # Fast smoothing via rolling sum of outer products
    if win and win > 0:
        outers = np.einsum('ni,nj->nij', quat_corr, quat_corr)
        prefix = np.cumsum(outers, axis=0)

        for i in range(len(quat_corr)):
            M = prefix[i] - (prefix[i - win - 1] if i - win - 1 >= 0 else 0)
            _, V = np.linalg.eigh(M)
            quat_corr_sm[i] = V[:, -1] if V[0, -1] > 0 else -V[:, -1]

    # Apply correction
    quat_out = np.conj(quat_tor) * qc.array(quat_corr_sm).normalized * quat_arm
    return quat_out
