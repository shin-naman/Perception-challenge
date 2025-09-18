# depth_xyz.py
import numpy as np

_WARNED_EXTRA_CH = False  # one-time warning per run

def load_xyz_npz(npz_path: str) -> np.ndarray:
    """
    Load camera-coordinate array from an NPZ.
    Prefers 'points' per challenge spec; falls back to 'xyz'.
    Accepts (H, W, 3) or (H, W, 4) and returns (H, W, 3).
    """
    global _WARNED_EXTRA_CH
    with np.load(npz_path) as data:
        key = "points" if "points" in data.files else ("xyz" if "xyz" in data.files else None)
        if key is None:
            raise ValueError(
                f"{npz_path}: expected key 'points' (per spec) or 'xyz'. "
                f"Available keys: {list(data.files)}"
            )
        arr = data[key]
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(f"{npz_path}: expected 3D array, got {getattr(arr, 'shape', None)}.")

        # Accept (H,W,3) or (H,W,4) -> return (H,W,3)
        if arr.shape[-1] == 3:
            xyz = arr.astype(float)
        elif arr.shape[-1] == 4:
            xyz = arr[..., :3].astype(float)  # first three channels are X,Y,Z
            if not _WARNED_EXTRA_CH:
                print("[WARN] NPZ has 4 channels; using first 3 as (X,Y,Z).", flush=True)
                _WARNED_EXTRA_CH = True
        else:
            raise ValueError(
                f"{npz_path}: expected last dim 3 or 4, got {arr.shape[-1]} (shape {arr.shape})."
            )

        # Final sanity: (H,W,3)
        if xyz.shape[-1] != 3:
            raise ValueError(f"{npz_path}: after processing, not (H,W,3): {xyz.shape}")
        return xyz

def robust_xyz_at(xyz: np.ndarray, u: int, v: int, k: int = 5) -> tuple[float, float, float]:
    """
    Sample a kxk patch around (u,v) from xyz (H,W,3) and return median (X,Y,Z).
    Filters NaNs and zero vectors; falls back to single pixel.
    """
    H, W, _ = xyz.shape
    u0 = int(np.clip(u, 0, W - 1))
    v0 = int(np.clip(v, 0, H - 1))

    half = max(1, k // 2)
    u1, u2 = max(0, u0 - half), min(W, u0 + half + 1)
    v1, v2 = max(0, v0 - half), min(H, v0 + half + 1)

    patch = xyz[v1:v2, u1:u2, :]  # (h, w, 3)
    pts = patch.reshape(-1, 3)

    finite = np.isfinite(pts).all(axis=1)
    nonzero = np.linalg.norm(pts, axis=1) > 0
    good = finite & nonzero
    pts = pts[good]

    if pts.size == 0:
        X, Y, Z = xyz[v0, u0]
        return float(X), float(Y), float(Z)

    X, Y, Z = np.median(pts, axis=0)
    return float(X), float(Y), float(Z)

__all__ = ["load_xyz_npz", "robust_xyz_at"]
