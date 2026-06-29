import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import math

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IRSIM_DIR = os.path.join(BASE_DIR, "IRSimDataset")
GRID_DIR = os.path.join(BASE_DIR, "OccupancyGrids")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_grid_alignment")

os.makedirs(DEBUG_DIR, exist_ok=True)

# Find all grid files
grid_files = [f for f in os.listdir(GRID_DIR) if f.endswith(".npy")]

for g_file in grid_files:
    map_id = g_file.replace(".npy", "")
    
    #Load Grid and Meta
    grid = np.load(os.path.join(GRID_DIR, g_file))
    with open(os.path.join(GRID_DIR, f"{map_id}_meta.json"), 'r') as f:
        meta = json.load(f)
    
    #Load Corresponding YAML (handle world_ prefix if necessary)
    yaml_name = f"{map_id}.yaml"
    yaml_path = os.path.join(IRSIM_DIR, yaml_name)
    
    if not os.path.exists(yaml_path):
        print(f"Skipping {map_id}: YAML not found at {yaml_path}")
        continue

    with open(yaml_path, 'r') as f:
        world = yaml.safe_load(f)
    obstacles = world.get("obstacle", [])

    #Create Plot
    plt.figure(figsize=(12, 10))
    
    # We display the grid. 
    # extent=[left, right, bottom, top] maps the pixel indices to world coordinates
    # Using the meta math: x_world = (col * res) + min_x - margin
    left = meta['min_x'] - meta['margin']
    right = left + (grid.shape[1] * meta['resolution'])
    bottom = meta['min_y'] - meta['margin']
    top = bottom + (grid.shape[0] * meta['resolution'])
    
    # Display the occupancy grid (0=white, 1=black)
    # cmap='gray_r' makes 0=white and 1=black
    plt.imshow(grid, cmap='gray_r', origin='lower', extent=[left, right, bottom, top])

    # 4. Overlay IRSim Obstacles
    for obs in obstacles:
        x, y, theta = obs["state"]
        l = obs["shape"]["length"]
        w = obs["shape"]["width"]

        dx, dy = l/2, w/2
        corners = [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]
        
        corners_rot = [
            [
                x + cx * math.cos(theta) - cy * math.sin(theta),
                y + cx * math.sin(theta) + cy * math.cos(theta)
            ]
            for cx, cy in corners
        ]
        
        cx, cy = zip(*corners_rot)
        # We use a semi-transparent bright color to see the overlap
        plt.fill(cx + (cx[0],), cy + (cy[0],), color='orange', alpha=0.5, edgecolor='red', label="IRSim Wall")

    plt.title(f"Alignment Check: {map_id}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the debug image
    save_path = os.path.join(DEBUG_DIR, f"align_{map_id}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Verified {map_id} -> Saved to {save_path}")

print(f"\nCheck complete. Images saved in: {DEBUG_DIR}")