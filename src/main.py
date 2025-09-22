# src/main.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from tracking import centers_from_csv
from depth_xyz import load_xyz, robust_xyz_at
from geometry import align_and_negate
from plotting import save_traj_png


def frame_id_to_name(fid: int) -> str:
    """
    Map CSV frame_id to NPZ filename stem.

    Your dataset uses names like: depth000000.npz
    """
    return f"depth{fid:06d}"


def collect_camera_to_light_vectors(
    dataset_dir: Path,
    top_bias: float = 0.35,
    k_patch: int = 5,
) -> np.ndarray:
    """
    Build the (N,2) sequence of camera→light ground-plane vectors (X,Y) across frames.
    """
    csv_path = dataset_dir / "bboxes_light.csv"
    xyz_dir = dataset_dir / "xyz"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path.resolve()}")
    if not xyz_dir.exists():
        raise FileNotFoundError(f"Missing xyz directory: {xyz_dir.resolve()}")

    centers = centers_from_csv(csv_path, top_bias=top_bias)

    p_cam: List[Tuple[float, float]] = []
    used = 0
    skipped_missing = 0

    for fid in tqdm(sorted(centers.keys()), desc="Sampling XYZ", unit="frame"):
        stem = frame_id_to_name(fid)
        npz_path = xyz_dir / f"{stem}.npz"
        if not npz_path.exists():
            skipped_missing += 1
            continue

        u, v = centers[fid]
        xyz = load_xyz(npz_path)
        X, Y, Z = robust_xyz_at(xyz, u=u, v=v, k=k_patch)
        p_cam.append((X, Y))
        used += 1

    if used == 0:
        raise RuntimeError("No frames processed — check file naming and CSV frame_ids.")

    p_cam_arr = np.asarray(p_cam, dtype=float)  # (N,2)

    # Quick sanity: distance to light tends to decrease
    d0, d1 = np.linalg.norm(p_cam_arr[0]), np.linalg.norm(p_cam_arr[-1])
    print(f"[sanity] ||camera→light|| start≈{d0:.2f} m → end≈{d1:.2f} m (should generally decrease)")
    print(f"[stats] frames used: {used}, missing xyz files skipped: {skipped_missing}")

    return p_cam_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Ego trajectory from bboxes + depth")
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset",
        help="Path to dataset dir containing bboxes_light.csv and xyz/",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Also write outputs/trajectory.mp4 animation",
    )
    parser.add_argument(
        "--top-bias",
        type=float,
        default=0.35,
        help="Top bias [0,1] for picking pixel inside bbox (default 0.35).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Odd patch size for robust depth sampling (default 5).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.data)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect camera→light vectors (X,Y) per frame
    p_cam = collect_camera_to_light_vectors(
        dataset_dir=dataset_dir,
        top_bias=args.top_bias,
        k_patch=args.k,
    )

    # 2) Geometry: rotate by −θ0 and negate → ego positions (x=Forward, y=Lateral)
    traj_xy = align_and_negate(p_cam)

    # 3) Plot
    png_path = outputs_dir / "trajectory.png"
    save_traj_png(traj_xy, png_path, title="Ego Trajectory (BEV)")

    # Final summary
    end_dist = np.linalg.norm(traj_xy[-1])
    print(f"[done] wrote {png_path}")
    print(f"[summary] end distance to origin ≈ {end_dist:.2f} m")


if __name__ == "__main__":
    main()
