import os
import yaml
import matplotlib.pyplot as plt
import math


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.join(BASE_DIR, "IRSimDataset_Sorted", "Hard")
DEBUG_DIR = os.path.join(BASE_DIR, "hard_plots_debug")

os.makedirs(DEBUG_DIR, exist_ok=True)

# Gather all YAML files directly from the TestDataset folder
files = [f for f in os.listdir(TEST_ROOT) if f.endswith(".yaml") or f.endswith(".yml")]

if not files:
    print(f"CRITICAL: No .yaml or .yml files found directly in {TEST_ROOT}!")
    exit()

print(f"Found {len(files)} YAML maps in TestDataset. Generating plots...")


for idx, file in enumerate(files, 1):
    yaml_path = os.path.join(TEST_ROOT, file)
    
    # Load the IRSim YAML directly
    with open(yaml_path, "r") as f:
        world = yaml.safe_load(f)
    
    # Grab obstacles (default to empty list if none exist in the file)
    obstacles = world.get("obstacle", [])

    # Initialize Figure
    plt.figure(figsize=(10, 8))
    
    obstacle_labeled = False
    for obs in obstacles:
        x, y, theta = obs["state"]
        l = obs["shape"]["length"]
        w = obs["shape"]["width"]

        dx = l / 2
        dy = w / 2
        
        # Local coordinate corners of the rectangle
        corners = [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ]
        
        # Rotate and translate corners to world coordinates
        corners_rot = [
            [
                x + cx * math.cos(theta) - cy * math.sin(theta),
                y + cx * math.sin(theta) + cy * math.cos(theta)
            ]
            for cx, cy in corners
        ]
        
        cx, cy = zip(*corners_rot)
        
        # Single label for the legend to keep it clean
        label = "IRSim Obstacles" if not obstacle_labeled else ""
        plt.fill(cx, cy, alpha=0.5, color='orange', edgecolor='darkorange', label=label)
        obstacle_labeled = True

    # Styling and Rendering settings
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if obstacle_labeled:
        plt.legend(loc="upper right")
        
    plt.title(f"Map Preview: {file}", fontsize=12, fontweight='bold')
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    
    # Save using the base filename but switching extension to .png
    output_filename = os.path.splitext(file)[0] + ".png"
    plt.savefig(os.path.join(DEBUG_DIR, output_filename), dpi=150, bbox_inches='tight')
    plt.close()

    if idx % 10 == 0 or idx == len(files):
        print(f"Progress: [{idx}/{len(files)}] maps plotted.")

print(f"\nSuccess! Rendered images saved to: {DEBUG_DIR}")