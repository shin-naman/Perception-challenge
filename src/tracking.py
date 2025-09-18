# tracking.py
import pandas as pd

REQUIRED = ["frame_id", "x_min", "y_min", "x_max", "y_max"]
_TOP_BIAS = 0.35  # nudge the pixel up within the bbox (lights sit near the top)

def load_bboxes(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Validate expected columns
    if not all(c in df.columns for c in REQUIRED):
        raise ValueError(
            f"Missing bbox columns. Need: {REQUIRED}. Got: {list(df.columns)}"
        )

    # Coerce numeric types
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
    for c in ["x_min", "y_min", "x_max", "y_max"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop bad/empty rows
    df = df.dropna(subset=["frame_id", "x_min", "y_min", "x_max", "y_max"])

    # Remove rows where the bbox is 0,0,0,0 (no detection)
    no_box = (df["x_min"] == 0) & (df["y_min"] == 0) & (df["x_max"] == 0) & (df["y_max"] == 0)
    df = df[~no_box].copy()

    # Normalize types and ordering
    df["frame_id"] = df["frame_id"].astype(int)
    df = df.sort_values("frame_id").reset_index(drop=True)
    return df

def bbox_center(row) -> tuple[int, int]:
    # top-biased vertical center helps when the light is near the top of its box
    u = int(round((row.x_min + row.x_max) / 2.0))
    v = int(round(row.y_min + _TOP_BIAS * (row.y_max - row.y_min)))
    return u, v

def centers_from_csv(csv_path: str) -> dict[int, tuple[int, int]]:
    df = load_bboxes(csv_path)
    centers: dict[int, tuple[int, int]] = {}
    for _, r in df.iterrows():
        centers[int(r.frame_id)] = bbox_center(r)
    return centers
