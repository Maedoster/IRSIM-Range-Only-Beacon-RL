import os
import yaml
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import irsim
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv

# ==========================================
# 1. Directory Setup & Helpers
# ==========================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MAP_DIR = os.path.join(
    PROJECT_ROOT,
    "HouseExpo_IRSim_Converter",
    "IRSimDataset"
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WORLDS_DIR, exist_ok=True)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def choose_map():
    maps = [f for f in os.listdir(MAP_DIR) if f.endswith(".yaml")]
    if not maps:
        print(f"No maps found in {MAP_DIR}")
        exit(1)

    print("\nAvailable maps:")
    for i, m in enumerate(maps):
        print(f"{i}: {m}")

    idx = int(input("\nSelect map index: "))
    return os.path.join(MAP_DIR, maps[idx]), maps[idx]

def get_position(name):
    x = float(input(f"{name} x: "))
    y = float(input(f"{name} y: "))
    return [x, y, 0]  # Default theta to 0 for simplicity

def get_goal():
    x = float(input("Goal x: "))
    y = float(input("Goal y: "))
    return [x, y]


# ==========================================
# 2. Gymnasium Environment definition
# ==========================================

class RobotNavEnv(gym.Env):
    def __init__(self, render=False, world_file="robot_world.yaml"):
        super(RobotNavEnv, self).__init__()
        
        self.render_mode = render
        self.state_dim = 49  
        self.max_steps = 150  
        
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]),  
            high=np.array([0.6,  1.2]),   
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize simulator
        self.sim = irsim.make(world_file, display=render)
        self._reset_episode_tracking()

    def _reset_episode_tracking(self):
        self.time = 0
        self.last_position = None
        self.total_distance = 0
        self.total_velocity = 0

    def _calculate_metrics(self, current_position, action):
        if self.last_position is not None:
            step_distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += step_distance
            self.total_velocity += np.linalg.norm(action)
        self.last_position = current_position

    def _get_episode_info(self, terminal, reward):
        avg_velocity = self.total_velocity / self.time if self.time > 0 else 0
        return {
            'success': terminal and reward > 0,
            'collision': terminal and reward < 0,
            'total_distance': self.total_distance,
            'average_velocity': avg_velocity
        }

    def _extract_sim_data(self, action):
        scan = self.sim.get_lidar_scan()
        latest_scan = scan["ranges"] if isinstance(scan, dict) else scan
        
        robot_state = self.sim.get_robot_state()  
        
        # Handle IRSim object structure variation
        if hasattr(self.sim, 'robot'):
            robot_goal = self.sim.robot.goal
            collision = self.sim.robot.collision
            goal = self.sim.robot.arrive
        else:
            info = self.sim.get_robot_info(0)
            robot_goal = info.goal
            collision = info.collision
            goal = info.arrive
        
        goal_vector = [robot_goal[0,0] - robot_state[0,0], robot_goal[1,0] - robot_state[1,0]]
        distance = np.linalg.norm(goal_vector)
        
        pose_vector = [np.cos(robot_state[2,0]), np.sin(robot_state[2,0])]
        norm_pose = pose_vector / (np.linalg.norm(pose_vector) + 1e-6)
        norm_goal = goal_vector / (distance + 1e-6)
        
        cos_val = np.dot(norm_pose, norm_goal)
        sin_val = np.cross(norm_pose, norm_goal)
        diff_rad = np.arccos(np.clip(cos_val, -1.0, 1.0))
        
        return latest_scan, distance, cos_val, sin_val, collision, goal, diff_rad, action

    def reward(self, data):
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action = data
        reward = 0
        if collision:
            reward = -100
        elif goal:
            reward = 100
        else:
            reward = -distance * 10 + cos * 5 - abs(diff_rad) * 2 - np.linalg.norm(action) * 0.1

        min_lidar = np.min(latest_scan)
        if min_lidar < 0.3:
            reward -= 0.5  # Penalty for being too close to walls

        return latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward

    def prepare_state(self, data):
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        scan_arr = np.array(latest_scan)
        scan_arr[np.isinf(scan_arr)] = 10

        max_bins = self.state_dim - 7
        bin_size = len(scan_arr) // max_bins
        min_values = [np.min(scan_arr[i:i+bin_size])/10 for i in range(0, max_bins * bin_size, bin_size)]

        state = min_values + [distance/10, cos, sin, (action[0]+0.6)/1.2, (action[1]+1.2)/2.4, np.cos(diff_rad), np.sin(diff_rad)]
        return np.array(state, dtype=np.float32), (collision or goal)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim.reset()
        sim_data = self._extract_sim_data(action=[0.0, 0.0])
        sim_data = self.reward(sim_data)
        obs, _ = self.prepare_state(sim_data)
        
        self._reset_episode_tracking()
        return obs, {}

    def step(self, action):
        ctrl_action = np.array([[-action[0]], [action[1]]])
        self.sim.step(ctrl_action)
        
        sim_data = self._extract_sim_data(action=action)
        sim_data = self.reward(sim_data)
        obs, terminal = self.prepare_state(sim_data)
        reward = sim_data[-1]

        self._calculate_metrics(self.sim.get_robot_state()[:2], action)
        self.time += 1
        
        terminated = bool(terminal)
        truncated = self.time >= self.max_steps
        
        if truncated and not terminated:
            reward = -100

        return obs, float(reward), terminated, truncated, self._get_episode_info(terminated, reward)

def make_env(render=False, world_file="robot_world.yaml"):
    def _init():
        return RobotNavEnv(render=render, world_file=world_file)
    return _init


# ==========================================
# 3. Main Logic (Map Selection -> Training)
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=4) 
    parser.add_argument('--total-timesteps', type=int, default=200000)
    args = parser.parse_args()

    # 1. Select map and get start/goal
    map_path, map_file = choose_map()
    map_name = os.path.splitext(map_file)[0]
    world_data = load_yaml(map_path)

    start = get_position("Start")
    goal = get_goal()

    # 2. Add robot dynamically to the map data
    world_data["robot"] = [{
        "kinematics": {"name": "diff"},
        "shape": {"name": "circle", "radius": 0.2},
        "vel_min": [-2, -2],
        "vel_max": [2, 2],
        "state": start,  
        "goal": goal + [0], 
        "arrive_mode": "state",
        "goal_threshold": 0.2,
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

    # 3. Save the temporary world file to disk
    temp_path = os.path.join(WORLDS_DIR, f"world_{map_name}.yaml")
    with open(temp_path, "w") as f:
        yaml.dump(world_data, f, sort_keys=False)
    
    print(f"\nSaved environment configuration to: {temp_path}")

    # 4. Initialize environments for SB3 using the generated file
    print(f"Initializing {args.num_envs} environments...")
    env = SubprocVecEnv([make_env(world_file=temp_path) for _ in range(args.num_envs)])

    # 5. Initialize TD3 and begin training
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", 
        batch_size=128, 
        buffer_size=10000, 
        tensorboard_log="./logs/"
    )
        
    print("\nStarting Training...")
    model.learn(total_timesteps=args.total_timesteps)
        
    # 6. Save Model
    save_path = os.path.join(MODELS_DIR, f"td3_robot_nav_{map_name}")
    model.save(save_path)
    print(f"\nTraining Complete. Model saved to {save_path}.zip")

    

if __name__ == '__main__':
    main()