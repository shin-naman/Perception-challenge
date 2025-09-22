# Perception Coding Challenge

## Method

I approached this project by first using GPT to help me understand the given dataset and how its components intersect. The dataset provided two main inputs:  
- A CSV file (`bboxes_light.csv`) containing bounding boxes for the traffic light in each frame.  
- Depth data (`.npz` files) containing 3D coordinates `(X, Y, Z)` for every pixel in the camera image.  

The task was to use these inputs to estimate the ego vehicle’s trajectory relative to the traffic light.

Initially, I sampled the raw `(X, Y)` coordinates from the depth data at the bounding box center and plotted them. The resulting graph looked completely wrong — the trajectory didn’t line up with expectations. With GPT’s help, I realized that the vectors described the camera→light direction, while I needed the ego (camera) position relative to a fixed light. This required **negating the vectors**.  

After applying negation, the plot improved but was slanted. The fix was to **rotate all vectors** so that the first one aligned with the forward axis. With rotation and negation combined, the trajectory matched expectations and resembled the sample solution.

---

## Implementation

To keep the project organized, I separated the logic into modular files:

- **`tracking.py`**  
  Reads the CSV, skips invalid rows, and computes the bounding box center `(u,v)` (pixel coordinates). This locates the traffic light in each frame.

- **`depth_xyz.py`**  
  Loads each `.npz` file as an `(H, W, 3)` array of `(X, Y, Z)` camera coordinates.  
  From the `(u,v)` pixel, it samples a robust `(X,Y,Z)` using a small patch median to reduce noise.

- **`geometry.py`**  
  Converts camera→light vectors into ego positions.  
  - Compute the first vector’s angle.  
  - Rotate all vectors so the first aligns with +X (forward).  
  - Negate to switch from camera→light to light→camera.  
  This produces a sequence of `(x_forward, y_lateral)` points for plotting.

- **`plotting.py`**  
  Plots the trajectory, marking the start, end, and traffic light at the origin. Ensures the origin is visible and outputs `trajectory.png`.

- **`main.py`**  
  Orchestrates the full pipeline: load data, compute trajectory, and save the plot.

---

## Assumptions

- The CSV’s `frame_id` matches the depth `.npz` filenames (renamed as `depth%06d.npz`).  
- The traffic light is always visible and correctly annotated in the CSV.  

---

## Results

The final trajectory plot shows the ego vehicle approaching the fixed traffic light at the origin. The path begins far away (forward ≈ –40 m) and converges smoothly toward `(0,0)`. The modular structure makes it easy to adjust parameters (e.g., patch size, top-bias in bounding boxes) without changing the overall pipeline. Graph is not rotated to match sample plot, but the trajectory is very similar. 

