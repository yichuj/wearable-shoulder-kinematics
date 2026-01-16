import numpy as np
import quaternionic as qc
import matplotlib.pyplot as plt


def rms(err):
    """
    Compute the root-mean-square (RMSE) of an error signal.

    NaN values are ignored.

    Parameters
    ----------
    err : array-like
        Error values over time.

    Returns
    -------
    float
        Root-mean-square error.
    """
    return np.sqrt(np.nanmean(err ** 2))


def get_err(q1, q2):
    """
    Compute overall orientation error between two quaternion sequences.

    The error at each time frame is defined as the rotation angle of the
    relative orientation between the estimated and reference quaternions,
    corresponding to the axis–angle representation.

    Parameters
    ----------
    q1 : array-like, shape (N, 4)
        Estimated orientation quaternions in (w, x, y, z) order.
    q2 : array-like, shape (N, 4)
        Reference (ground-truth) orientation quaternions in (w, x, y, z) order.

    Returns
    -------
    np.ndarray, shape (N,)
        Overall orientation error in degrees at each time frame.
    """
    
    q1 = np.ascontiguousarray(np.asarray(q1, dtype=np.float64))
    q2 = np.ascontiguousarray(np.asarray(q2, dtype=np.float64))

    q1 = qc.array(q1).normalized
    q2 = qc.array(q2).normalized
    return np.degrees(qc.distance.rotation.intrinsic(q1, q2))


def get_all_err(t, sig1, sig2, OMC_idx, tag=""):
    """
    Compute and visualize quaternion angular error across trials.

    Overall orientation error is computed at each time frame as the
    axis–angle magnitude of the relative rotation between the estimated
    and reference quaternions. Errors are evaluated segment-wise using
    trial index ranges and summarized using RMSE across time.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time vector in seconds.
    sig1 : array-like, shape (N, 4)
        Reference quaternion signal in (w, x, y, z) order.
    sig2 : array-like, shape (N, 4)
        Estimated quaternion signal in (w, x, y, z) order.
    OMC_idx : list of tuple
        List of (start_idx, end_idx) pairs defining trial segments
        (end index exclusive).
    tag : str, optional
        Label used for console output and figure filename.

    Returns
    -------
    np.ndarray, shape (N,)
        Full-length angular error signal in degrees, with NaNs outside
        trial segments.
    """

    # calculate & print errors
    print(f"{tag}")
    print("  RMSE for each trial:", end=" ")
    err_full = np.full((sig1.shape[0],), np.nan)
    for i0, iN in OMC_idx:
        err = get_err(sig1[i0:iN,:], sig2[i0:iN,:])
        err_full[i0:iN] = err
        print(f"{rms(err):.1f}\N{DEGREE SIGN}", end=", ")
    print(f"\n  Overall RMSE: {rms(err_full):.1f}\N{DEGREE SIGN}")
    
    # plot errors
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize = (8, 1))
    plt.plot(t, err_full, linewidth=1)
    plt.axhline(rms(err_full), color='r', linestyle='--', linewidth=0.75, label="Overall RMSE")
    plt.xlabel("Time (s)"), plt.ylabel("Error (\N{DEGREE SIGN})")
    plt.ylim(-2, 50), plt.xlim(0, max(t))
    plt.legend(loc='upper right')
    plt.title(f"Time-Series Error Plot Using {tag}")
    plt.show()
        
    return err_full
