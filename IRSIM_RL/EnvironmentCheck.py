import os
import tempfile
import yaml
import irsim

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up to project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MAP_DIR = os.path.join(
    PROJECT_ROOT,
    "HouseExpo_IRSim_Converter",
    "IRSimDataset"
)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def choose_map():
    maps = [f for f in os.listdir(MAP_DIR) if f.endswith(".yaml")]

    print("\nAvailable maps:")
    for i, m in enumerate(maps):
        print(f"{i}: {m}")

    idx = int(input("\nSelect map index: "))
    return os.path.join(MAP_DIR, maps[idx])


def get_position(name):
    x = float(input(f"{name} x: "))
    y = float(input(f"{name} y: "))
    theta = float(input(f"{name} theta (rad, default 0): ") or 0)
    return [x, y, theta]


def get_goal():
    x = float(input("Goal x: "))
    y = float(input("Goal y: "))
    return [x, y]

def run_sim(world_data):
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as f:
        yaml.dump(world_data, f, sort_keys=False)
        temp_path = f.name

    # Pass path to IRSim
    env = irsim.make(temp_path)

    while True:
        env.step()
        env.render()


def main():
    #Select map
    map_path = choose_map()
    world_data = load_yaml(map_path)

    #Get robot start + goal
    start = get_position("Start")
    goal = get_goal()

    #Add robot dynamically
    world_data["robot"] = [{
    "kinematics": {"name": "diff"},
    "shape": {"name": "circle", "radius": 0.2},
    "vel_min": [-2, -2],
    "vel_max": [2, 2],
    "state": start + [0],  # if start is [x, y, theta], we append 0 for z
    "goal": goal + [0],    # similarly append 0
    "arrive_mode": "state",
    "goal_threshold": 0.2,
    "behavior": {"name": "rvo"},

    "sensors": [
        {
            "type": "lidar2d",
            "range_min": 0,
            "range_max": 7,
            "angle_range": 6.28,
            "number": 420,
            "noise": True,
            "std": 0.08,
            "angle_std": 0.1,
            "offset": [0.15, 0, 0],
            "alpha": 0.3
        }
    ]
}]
    #Run simulation
    
    run_sim(world_data)




if __name__ == "__main__":
    main()