# main.py
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import sys

from tracking import centers_from_csv
from depth_xyz import load_xyz_npz, robust_xyz_at
from geometry import ego_trajectory_world
from plotting import save_trajectory_png, save_trajectory_mp4

def frame_id_to_name(fid: int) -> str:
    # Your depth files are named depth000000.npz, depth000001.npz, ...
    return f"depth{fid:06d}"

def main(make_video: bool, dataset_dir: Path):
    bbox_csv = dataset_dir / "bboxes_light.csv"
    xyz_dir  = dataset_dir / "xyz"

    # Helpful checks
    if not bbox_csv.exists():
        print(f"[ERROR] Missing CSV: {bbox_csv.resolve()}", file=sys.stderr)
        print("Hint: pass the correct folder with --data /path/to/dataset", file=sys.stderr)
        sys.exit(1)
    if not xyz_dir.exists():
        print(f"[ERROR] Missing XYZ dir: {xyz_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    centers = centers_from_csv(str(bbox_csv))

    p_cam_seq = []  # list of (X,Y) camera->light
    skipped = 0
    for fid in tqdm(sorted(centers.keys())):
        u, v = centers[fid]
        npz_path = xyz_dir / f"{frame_id_to_name(fid)}.npz"
        if not npz_path.exists():
            skipped += 1
            continue
        xyz = load_xyz_npz(str(npz_path))
        X, Y, Z = robust_xyz_at(xyz, u, v, k=5)
        p_cam_seq.append((X, Y))

    if not p_cam_seq:
        print("[ERROR] No frames processed. Check file names and CSV frame_ids.", file=sys.stderr)
        sys.exit(1)

    traj_xy = ego_trajectory_world(p_cam_seq)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_png(traj_xy, str(out_dir / "trajectory.png"))
    if make_video:
        save_trajectory_mp4(traj_xy, str(out_dir / "trajectory.mp4"), fps=10)

    if skipped:
        print(f"[WARN] Skipped {skipped} frame(s) with missing NPZ.", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset"),
                        help="Path to folder containing bboxes_light.csv, rgb/, xyz/")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 generation")
    args = parser.parse_args()
    main(make_video=not args.no_video, dataset_dir=args.data)
