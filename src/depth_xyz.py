# src/depth_xyz.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_xyz(npz_path: str | Path) -> np.ndarray:
    """
    Load a per-frame NPZ as an (H, W, 3) float array of camera coordinates (X,Y,Z).

    Accepts either key 'points' (preferred per spec) or 'xyz'.
    Accepts last-dim 3 or 4; if 4, returns first 3 channels.

    Returns
    -------
    xyz : np.ndarray
        Array of shape (H, W, 3) with dtype float.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path.resolve()}")

    with np.load(npz_path) as data:
        key = "points" if "points" in data.files else ("xyz" if "xyz" in data.files else None)
        if key is None:
            raise ValueError(
                f"{npz_path.name}: expected key 'points' or 'xyz'. "
                f"Available: {list(data.files)}"
            )
        arr = data[key]

    if arr.ndim != 3:
        raise ValueError(f"{npz_path.name}: expected 3D array, got shape {arr.shape}")

    # Handle (H,W,3) or (H,W,4) → return (H,W,3)
    if arr.shape[-1] == 3:
        xyz = arr
    elif arr.shape[-1] == 4:
        xyz = arr[..., :3]
    else:
        raise ValueError(
            f"{npz_path.name}: last dimension must be 3 or 4; got {arr.shape[-1]}"
        )

    return xyz.astype(float, copy=False)


def robust_xyz_at(xyz: np.ndarray, u: int, v: int, k: int = 5) -> Tuple[float, float, float]:
    """
    Sample a robust (X,Y,Z) at pixel (u,v) using a k×k patch median.

    Steps:
      - Clip (u,v) to image bounds
      - Extract k×k patch around (u,v)
      - Filter out rows with NaNs or zero vectors
      - Return median (X,Y,Z); fall back to single pixel if no good samples

    Parameters
    ----------
    xyz : np.ndarray
        (H, W, 3) array of camera coordinates.
    u, v : int
        Pixel column (u) and row (v). Note: depth indexing is [v, u].
    k : int
        Odd patch size (default 5). Try 3 or 7 if needed.

    Returns
    -------
    (X, Y, Z) : tuple[float, float, float]
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (H, W, 3); got {xyz.shape}")

    H, W, _ = xyz.shape
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    # patch bounds
    half = max(1, k // 2)
    u1, u2 = max(0, u - half), min(W, u + half + 1)
    v1, v2 = max(0, v - half), min(H, v + half + 1)

    patch = xyz[v1:v2, u1:u2, :].reshape(-1, 3)  # (N,3)

    # filter invalid/zero points
    finite = np.isfinite(patch).all(axis=1)
    nonzero = np.linalg.norm(patch, axis=1) > 0.0
    good = patch[finite & nonzero]

    if good.size == 0:
        X, Y, Z = xyz[v, u]
    else:
        X, Y, Z = np.median(good, axis=0)

    return float(X), float(Y), float(Z)


__all__ = ["load_xyz", "robust_xyz_at"]
