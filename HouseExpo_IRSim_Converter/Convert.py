import os
import json
import math
import yaml

WALL_THICKNESS = 0.1
MARGIN = 1.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "OriginalDataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "IRSimDataset")

def load_layout(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return min(xs), min(ys), max(xs), max(ys)

def shift_verts(verts, dx, dy):
    return [[x + dx, y + dy] for x, y in verts]

def edge_to_rectangle(p1, p2, thickness):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    angle = math.atan2(dy, dx)

    return {
        "kinematics": {"name": "static"},
        "shape": {
            "name": "rectangle",
            "length": length,
            "width": thickness
        },
        "state": [center_x, center_y, angle]
    }

def build_walls(verts):
    walls = []
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        walls.append(edge_to_rectangle(p1, p2, WALL_THICKNESS))
    return walls

def convert(layout):
    verts = layout["verts"]
    min_x, min_y, max_x, max_y = compute_bbox(verts)

    # Calculate World Dimensions
    width = (max_x - min_x) + 2 * MARGIN
    height = (max_y - min_y) + 2 * MARGIN

    # Shift vertices to positive coordinates starting from MARGIN
    shifted = shift_verts(verts, -min_x + MARGIN, -min_y + MARGIN)
    walls = build_walls(shifted)

    # Configuration for the Robot
    robot_config = [{ 
        "kinematics": {"name": "diff"},
        "shape": {"name": "circle", "radius": 0.15},
        "vel_min": [0, -1.2],
        "vel_max": [0.6, 1.2],
        "state": [1.0, 1.0, 0.0],  # Default start position
        "goal": [2.0, 2.0, 0.0],   # Default goal position
        "arrive_mode": "state",
        "goal_threshold": 0.2,
        "sensors": [
            {
                "type": "lidar2d",
                "range_min": 0,
                "range_max": 7,
                "angle_range": 6.28,
                "number": 100,
                "noise": True,
                "std": 0.05,
                "angle_std": 0.002,
                "offset": [0, 0, 0],
                "alpha": 0.3
            }
        ]
    }]

    return {
        "world": {
            "width": round(width, 2),
            "height": round(height, 2),
            "step_time": 0.1,
            "sample_time": 0.1,
            "offset": [0, 0]
        },
        "robot": robot_config,
        "obstacle": walls
    }

def save_yaml(data, path):
    with open(path, "w") as f:
        # sort_keys=False is key to keep the structure readable
        yaml.dump(data, f, sort_keys=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for file in files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file.replace(".json", ".yaml"))

        try:
            layout = load_layout(input_path)
            world_data = convert(layout)
            save_yaml(world_data, output_path)
            print(f"Successfully converted: {file}")
        except Exception as e:
            print(f"Error converting {file}: {e}")

    print(f"\nDone. Processed {len(files)} files.")

if __name__ == "__main__":
    main()