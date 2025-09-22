# src/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_traj_png(traj_xy: np.ndarray, out_path: str | Path, title: Optional[str] = None) -> None:
    """
    Save a clean Forward-vs-Lateral trajectory plot (PNG only).

    Parameters
    ----------
    traj_xy : (N,2) array of ego positions (x=Forward, y=Lateral)
    out_path : target PNG path (e.g., 'outputs/trajectory.png')
    title : optional figure title
    """
    traj_xy = np.asarray(traj_xy, dtype=float)
    if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
        raise ValueError(f"traj_xy must be (N,2); got {traj_xy.shape}")

    x = traj_xy[:, 0]  # Forward
    y = traj_xy[:, 1]  # Lateral

    _ensure_parent(out_path)

    # Bigger canvas so the data and origin fit comfortably
    plt.figure(figsize=(8, 8))

    # Trajectory line and key markers
    plt.plot(x, y, linewidth=2.5, label="Ego trajectory")
    plt.scatter([x[0]], [y[0]], marker="x", s=120, label="Start")
    plt.scatter([x[-1]], [y[-1]], s=80, label="End")
    plt.scatter([0], [0], marker="*", s=180, label="Traffic light (origin)")

    # Equal aspect and clean styling
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("Forward (m)")
    plt.ylabel("Lateral (m)")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ---- Bounds: ensure (0,0) is visible and add padding ----
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # Include origin in bounds
    xmin = min(xmin, 0.0)
    xmax = max(xmax, 0.0)
    ymin = min(ymin, 0.0)
    ymax = max(ymax, 0.0)

    # Add comfortable padding (~15%)
    dx = (xmax - xmin) * 0.15 + 1e-6
    dy = (ymax - ymin) * 0.15 + 1e-6
    plt.xlim(xmin - dx, xmax + dx)
    plt.ylim(ymin - dy, ymax + dy)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
