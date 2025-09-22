# src/tracking.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Columns we expect in the CSV
REQUIRED_COLUMNS = ["frame_id", "x_min", "y_min", "x_max", "y_max"]


def load_bboxes(csv_path: str | Path) -> pd.DataFrame:
    """
    Read and clean the bounding box CSV.

    Steps:
      - read CSV
      - ensure required columns exist
      - coerce to numeric and drop rows with NaNs in required columns
      - drop all-zero boxes (no detection)
      - fix inverted boxes (if x_max < x_min / y_max < y_min) by swapping
      - drop degenerate boxes (zero or negative width/height)
      - cast frame_id to int, sort by frame_id
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Bounding box CSV not found at: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # verify columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # coerce numeric
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with NaNs in required fields
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # drop all-zero boxes
    zero_box = (df.x_min == 0) & (df.y_min == 0) & (df.x_max == 0) & (df.y_max == 0)
    df = df.loc[~zero_box].copy()

    # fix inverted boxes by swapping min/max if needed
    x_swapped = df.x_max < df.x_min
    y_swapped = df.y_max < df.y_min
    if x_swapped.any():
        x_min_new = df.loc[x_swapped, "x_max"]
        x_max_new = df.loc[x_swapped, "x_min"]
        df.loc[x_swapped, "x_min"] = x_min_new
        df.loc[x_swapped, "x_max"] = x_max_new
    if y_swapped.any():
        y_min_new = df.loc[y_swapped, "y_max"]
        y_max_new = df.loc[y_swapped, "y_min"]
        df.loc[y_swapped, "y_min"] = y_min_new
        df.loc[y_swapped, "y_max"] = y_max_new

    # drop degenerate boxes (width/height <= 0 after fixes)
    width = df.x_max - df.x_min
    height = df.y_max - df.y_min
    valid = (width > 0) & (height > 0)
    df = df.loc[valid].copy()

    # normalize types and order
    df["frame_id"] = df["frame_id"].astype(int)
    df = df.sort_values("frame_id").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid bounding boxes remain after cleaning.")

    return df


def bbox_to_pixel(row: pd.Series, top_bias: float = 0.35) -> Tuple[int, int]:
    """
    Convert a single bbox row to a representative pixel (u, v).

    - u (column) = horizontal center of the box
    - v (row)    = top-biased vertical center (traffic lights are near top)
      v = y_min + top_bias * (y_max - y_min), with 0 <= top_bias <= 1

    Returns (u, v) as ints suitable for indexing into an image/depth array:
      depth[v, u]  (note: v is row, u is column)
    """
    if not (0.0 <= top_bias <= 1.0):
        raise ValueError(f"top_bias must be in [0,1], got {top_bias}")

    u = int(round((row.x_min + row.x_max) / 2.0))
    v = int(round(row.y_min + top_bias * (row.y_max - row.y_min)))
    return u, v


def centers_from_csv(csv_path: str | Path, top_bias: float = 0.35) -> Dict[int, Tuple[int, int]]:
    """
    Convenience: load CSV and produce {frame_id: (u, v)} for all rows.
    Later, you'll use these (u, v) to sample XYZ from the depth array.
    """
    df = load_bboxes(csv_path)
    centers: Dict[int, Tuple[int, int]] = {}
    for _, r in df.iterrows():
        centers[int(r.frame_id)] = bbox_to_pixel(r, top_bias=top_bias)
    return centers
