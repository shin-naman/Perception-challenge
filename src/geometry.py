# geometry.py
import numpy as np

def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def align_camera_to_world(p0_cam: np.ndarray) -> np.ndarray:
    """
    Rotate so the initial camera→light vector lies on +X.
    NOTE: no Y flip; world-Y matches camera-Y (right).
    """
    theta0 = np.arctan2(p0_cam[1], p0_cam[0])  # angle of p0 in camera plane
    R = rot2d(-theta0)                         # align p0 to +X
    F = np.eye(2)                              # no flip
    return F @ R

def ego_trajectory_world(p_cam_seq: list[tuple[float,float]]) -> np.ndarray:
    """
    p_cam_seq: list of (X_t, Y_t) camera→light vectors on ground.
    Returns an (N,2) array of ego (camera) positions in world coords.
    """
    p0 = np.array(p_cam_seq[0], dtype=float)
    A = align_camera_to_world(p0)
    traj = []
    for X, Y in p_cam_seq:
        p = np.array([X, Y], dtype=float)
        pw = A @ p
        C = -pw
        # --- swap axes here ---
        C = np.array([C[1], C[0]])  
        traj.append(C)
    return np.vstack(traj)
