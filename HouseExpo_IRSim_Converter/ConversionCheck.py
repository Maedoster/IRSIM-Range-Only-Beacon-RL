import os
import json
import yaml
import matplotlib.pyplot as plt
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE_DIR, "OriginalDataset")
IRSIM_DIR = os.path.join(BASE_DIR, "IRSimDataset")

# Make debug folder
DEBUG_DIR = os.path.join(BASE_DIR, "debug_plots")
os.makedirs(DEBUG_DIR, exist_ok=True)

files = [f for f in os.listdir(ORIGINAL_DIR) if f.endswith(".json")]

for file in files:
    # Load original JSON
    with open(os.path.join(ORIGINAL_DIR, file)) as f:
        layout = json.load(f)
    verts = layout["verts"]

    # Load corresponding IRSim YAML
    yaml_file = file.replace(".json", ".yaml")
    with open(os.path.join(IRSIM_DIR, yaml_file)) as f:
        world = yaml.safe_load(f)
    obstacles = world["obstacle"]

    # Plot original layout
    xs, ys = zip(*verts)
    plt.figure(figsize=(10, 8))
    plt.plot(xs + (xs[0],), ys + (ys[0],), 'b-', label="Original layout")

    # Plot IRSim rectangles
    for obs in obstacles:
        x, y, theta = obs["state"]
        l = obs["shape"]["length"]
        w = obs["shape"]["width"]

        dx = l/2
        dy = w/2
        corners = [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ]
        corners_rot = [
            [
                x + cx * math.cos(theta) - cy * math.sin(theta),
                y + cx * math.sin(theta) + cy * math.cos(theta)
            ]
            for cx, cy in corners
        ]
        cx, cy = zip(*corners_rot)
        plt.fill(cx, cy, alpha=0.3, color='orange')

    plt.axis('equal')
    plt.legend()
    plt.title(file)
    plt.savefig(os.path.join(DEBUG_DIR, file.replace(".json", ".png")))
    plt.close()

print(f"Done! Plots saved in {DEBUG_DIR}")