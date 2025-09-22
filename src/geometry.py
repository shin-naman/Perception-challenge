# src/geometry.py
from __future__ import annotations

import numpy as np


def rot2d(theta: float) -> np.ndarray:
    """
    Return the 2×2 rotation matrix R(theta).

    [x']   [ cosθ  -sinθ ] [x]
    [y'] = [ sinθ   cosθ ] [y]
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def align_and_negate(p_cam: np.ndarray) -> np.ndarray:
    """
    Convert camera→light ground-plane vectors into ego (camera) positions
    in an aligned world frame using only:
      1) rotation by −θ0 so the first vector lies on +X
      2) negation to switch from camera→light to light→camera

    Parameters
    ----------
    p_cam : np.ndarray
        (N, 2) array where each row is (X_t, Y_t) from the depth sampler.

    Returns
    -------
    traj_xy : np.ndarray
        (N, 2) array where each row is the ego position (x_t, y_t)
        after rotation and negation. No further mutations applied.
    """
    if p_cam is None:
        raise ValueError("p_cam is None; expected an (N,2) array.")
    p_cam = np.asarray(p_cam, dtype=float)
    if p_cam.ndim != 2 or p_cam.shape[1] != 2:
        raise ValueError(f"p_cam must have shape (N, 2); got {p_cam.shape}.")
    if p_cam.shape[0] == 0:
        raise ValueError("p_cam is empty; need at least one vector.")

    # First vector angle θ0 (camera→light at t=0)
    p0 = p_cam[0]
    norm0 = np.linalg.norm(p0)
    if not np.isfinite(norm0) or norm0 < 1e-9:
        raise ValueError("First vector has near-zero length or is invalid; cannot define θ0.")

    theta0 = np.arctan2(p0[1], p0[0])

    # Rotate all vectors by −θ0 so p0 aligns with +X
    R = rot2d(-theta0)
    p_aligned = (R @ p_cam.T).T  # (N,2)

    # Ego position is the negative of the aligned camera→light vector
    traj_xy = -p_aligned
    return traj_xy


__all__ = ["rot2d", "align_and_negate"]
