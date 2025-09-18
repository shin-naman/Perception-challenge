# plotting.py
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from pathlib import Path

def save_trajectory_png(traj_xy: np.ndarray, out_path: str):
    x, y = traj_xy[:,0], traj_xy[:,1]
    plt.figure(figsize=(6,6))
    plt.plot(x, y, linewidth=2, label="Ego trajectory")
    plt.scatter([x[0]],[y[0]], marker='x', s=120, label="Start")     # start
    plt.scatter([x[-1]],[y[-1]], s=80, label="End")                  # end
    plt.scatter([0], [0], marker='*', s=120, label="Traffic light (origin)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Forward (X, m)")
    plt.ylabel("Lateral (Y, m)")
    plt.title("Ego Trajectory (BEV)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_trajectory_mp4(traj_xy: np.ndarray, out_path: str, fps: int = 10):
    frames = []
    for i in range(1, len(traj_xy)+1):
        x, y = traj_xy[:i,0], traj_xy[:i,1]
        plt.figure(figsize=(6,6))
        plt.plot(x, y, linewidth=2, label="Ego trajectory")
        plt.scatter([x[0]],[y[0]], marker='x', s=120, label="Start")
        plt.scatter([x[-1]],[y[-1]], s=80, label="End")
        plt.scatter([0], [0], marker='*', s=120, label="Traffic light (origin)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Forward (X, m)")
        plt.ylabel("Lateral (Y, m)")
        plt.title(f"Ego Trajectory (frame {i}/{len(traj_xy)})")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.draw()
        fig = plt.gcf()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close()
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
