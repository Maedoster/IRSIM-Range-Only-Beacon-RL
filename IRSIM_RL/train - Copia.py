import os
import numpy as np
import yaml
import argparse

from robot_env import RobotNavEnv

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


# ==========================================
# Directory Setup & Helpers
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




def make_env(world_file):
    def _init():
        return RobotNavEnv(render=False, world_file=world_file)
    return _init


# ==========================================
# Main Logic (Map Selection -> Training)
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=8) 
    parser.add_argument('--total-timesteps', type=int, default=1000000) 
    args = parser.parse_args()

    # Select map and setup logic 
    map_path, map_file = choose_map()
    map_name = os.path.splitext(map_file)[0]
    world_data = load_yaml(map_path)

    start = get_position("Start")
    goal = get_goal()

    # Add robot dynamically to the map data
    world_data["robot"] = [{
        "kinematics": {"name": "diff"},
        "shape": {"name": "circle", "radius": 0.2},
        "vel_min": [0, -2],
        "vel_max": [2, 2],
        "state": start,  
        "goal": goal + [0], 
        "arrive_mode": "state",
        "goal_threshold": 0.2,
        "sensors": [
            {
                "type": "lidar2d",
                "range_min": 0,
                "range_max": 15,
                "angle_range": 6.28,
                "number": 100,
                "noise": True,
                "std": 0.08,
                "angle_std": 0.1,
                "offset": [0.15, 0, 0],
                "alpha": 0.3
            }
        ]
    }]

    # Save the temporary world file to disk
    temp_path = os.path.join(WORLDS_DIR, f"world_{map_name}.yaml")
    with open(temp_path, "w") as f:
        yaml.dump(world_data, f, sort_keys=False)

    # Create a single evaluation environment
    # We wrap it in 'Monitor' so it records the reward data for plots
    raw_env = RobotNavEnv(render=True, world_file=temp_path)
    monitored_env = Monitor(raw_env)
    eval_env = DummyVecEnv([lambda: monitored_env])

    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(MODELS_DIR, "best_model"),
        log_path=os.path.join(BASE_DIR, "logs", map_name),
        eval_freq=max(20000 // args.num_envs, 1), 
        n_eval_episodes=1,
        deterministic=True
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=os.path.join(MODELS_DIR, "crash_checkpoints"),
        name_prefix="td3_recovery"
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])


    # Initialize training environments
    print(f"Initializing {args.num_envs} environments...")
    
    env = SubprocVecEnv([make_env(temp_path) for _ in range(args.num_envs)])
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize TD3



    # 1. Define the Action Noise (Actor exploration noise: 0.5)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.5 * np.ones(n_actions)
    )

    # 2. Define the NN Architecture (Actor & Critic: [64, 32])
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 32], qf=[64, 32])
    )

    # 3. Initialize the Model
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        
        # Core Hyperparameters
        learning_rate=1e-3,          # SB3 uses one LR; usually set to Actor LR
        buffer_size=500000,          # Replay buffer size
        batch_size=32,               # Batch size (N)
        gamma=0.99,                  # Discount factor
        tau=0.01,                    # Target NN update rate
        
        # Training Frequency logic
        train_freq=30,               # Update every 30
        gradient_steps=20,           # Update times 20
        learning_starts=10000,       # Random start (Steps, not episodes)
        
        # TD3 Specifics
        policy_delay=2,              # Policy update delay
        action_noise=action_noise,   # Actor exploration noise
        
        # Architecture
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/"
    )

    
        
    print("\nStarting Training...")
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=callback_list  
    )
        
    # Save Final Model
    save_path = os.path.join(MODELS_DIR, f"td3_robot_nav_{map_name}_final")
    save_path_stats = os.path.join(MODELS_DIR, f"td3_robot_nav_{map_name}_final_stats")
    
    
    model.save(save_path)
    env.save(save_path_stats)

    print(f"\nTraining Complete. Best model saved in {MODELS_DIR}/best_{map_name}")

if __name__ == '__main__':
    main()