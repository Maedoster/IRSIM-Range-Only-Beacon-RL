import os
import json
import argparse
import random
import numpy as np

from robot_env import RobotNavEnv
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# ==========================================
# CONFIGURATION
# ==========================================
EXPERIMENT_DIR_NAME = "run_SAC_True_20360_86%" 
MODEL_FILE_NAME = "best_model" 

NUM_EPISODES = 50
INITIAL_SEED = np.random.randint(0, 100000) # Or a specific one for reproducibility, e.g., 12345
#INITIAL_SEED = 12345

# ==========================================
# Helpers
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPERIMENTS_DIR = os.path.join(MODELS_DIR, EXPERIMENT_DIR_NAME)

def load_config(experiment_folder):
    meta_path = os.path.join(experiment_folder, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    algo_str = meta["experiment_info"]["algorithm"]
    pf_active = meta["experiment_info"]["pf_active"]
    algo_map = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "DDPG": DDPG}
    return algo_map[algo_str], pf_active

# ==========================================
# Testing Execution
# ==========================================
def main():
    # 1. Detect Algorithm and PF status from training
    AlgoClass, pf_active = load_config(EXPERIMENTS_DIR)
    model_path = os.path.join(EXPERIMENTS_DIR, MODEL_FILE_NAME, "best_model.zip")
    stats_path = os.path.join(EXPERIMENTS_DIR, MODEL_FILE_NAME, "best_model.pkl")

    print("INITIAL SEED:", INITIAL_SEED)
    
    # 2. Create the Env ONE TIME
    def make_env():
        return RobotNavEnv(
            render=True, 
            pf_active=pf_active, 
            seed=INITIAL_SEED, 
            is_testing=True,
            is_eval=False
        )
    
    venv = DummyVecEnv([make_env])
    set_random_seed(INITIAL_SEED)

    # 3. Handle Normalization
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, venv)
        env.training = False
        env.norm_reward = False
    else:
        env = venv

    # 4. Load Model
    model = AlgoClass.load(model_path, env=env)

    print(f"Testing {AlgoClass.__name__} (PF={pf_active}) for {NUM_EPISODES} episodes...")

    # RESET ONCE BEFORE THE LOOP
    obs = env.reset() 
    raw_env = venv.envs[0] 
    
    ep = 0
    total_reward = 0
    print(f"\n--- Episode {ep+1} ---")
    print(f"Map: {raw_env.current_map_name}")

    # Use a single while loop to manage the vectorized transitions smoothly
    while ep < NUM_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        
        total_reward += reward[0]
        
        if dones[0]:  # Episode finished!
            info = infos[0]
            result = "SUCCESS" if info.get('success') else "FAILED"
            print(f"Result: {result} | Total Reward: {total_reward:.2f} | Steps: {info.get('steps')} | Error: {info.get('target_error'):.2f}")
            
            ep += 1
            total_reward = 0
            
            # If there are more episodes left, prepare the print statement
            if ep < NUM_EPISODES:
                print(f"\n--- Episode {ep+1} ---")
                # Because VecEnv auto-resets, raw_env has already loaded the next map internally!
                print(f"Map: {raw_env.current_map_name}")

    venv.close()

if __name__ == "__main__":
    main()