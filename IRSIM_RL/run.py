import os
import json
import argparse
import numpy as np

from robot_env import RobotNavEnv
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# ==========================================
# CONFIGURATION
# ==========================================
EXPERIMENT_DIR_NAME = "run_SAC_True_65807"
MODEL_FILE_NAME = "best_model" 

NUM_EPISODES = 50
INITIAL_SEED = np.random.randint(0, 100000) # Or a specific one for reproducibility, e.g., 12345
#INITIAL_SEED = 12345

# ==========================================
# Helpers
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_config(experiment_folder):
    meta_path = os.path.join(experiment_folder, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    algo_str = meta["experiment_info"]["algorithm"]
    pf_active = meta["experiment_info"]["pf_active"]
    #pf_active = False #If you want to override and select LS or PF
    algo_map = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "DDPG": DDPG}
    return algo_map[algo_str], pf_active

# ==========================================
# Testing Execution
# ==========================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", default=EXPERIMENT_DIR_NAME)
    parser.add_argument("--model", default=MODEL_FILE_NAME)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=INITIAL_SEED)

    args = parser.parse_args()
    experiments_dir = os.path.join(MODELS_DIR, args.experiment)
    # 1. Detect Algorithm and PF status from training
    AlgoClass, pf_active = load_config(experiments_dir)
    model_path = os.path.join(experiments_dir, args.model, "best_model.zip")
    stats_path = os.path.join(experiments_dir, args.model, "best_model.pkl")

    print("INITIAL SEED:", args.seed)
    
    # 2. Create the Env
    def make_env():
        return RobotNavEnv(
            render=True, 
            pf_active=pf_active, 
            seed=args.seed, 
            is_testing=False,
            is_eval=False,
            is_run=True
        )
    
    venv = DummyVecEnv([make_env])

    # 3. Handle Normalization
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, venv)
        env.training = False
        env.norm_reward = False
    else:
        env = venv

    # 4. Load Model
    model = AlgoClass.load(model_path, env=env)

    print(f"Testing {AlgoClass.__name__} (PF={pf_active}) for {args.episodes} episodes...")

    obs = env.reset()
    raw_env = venv.envs[0] 
    
    ep = 0
    total_reward = 0
    print(f"\n--- Episode {ep+1} ---")
    print(f"Map: {raw_env.current_map_name}")

    while ep < args.episodes:
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        
        total_reward += reward[0]
        
        if dones[0]:  # Episode finished
            info = infos[0]
            result = "SUCCESS" if info.get('success') else "FAILED"
            print(f"Result: {result} | Total Reward: {total_reward:.2f} | Steps: {info.get('steps')} | Error: {info.get('target_error'):.2f}")
            
            ep += 1
            total_reward = 0
            
            # If there are more episodes left, prepare the print statement
            if ep < args.episodes:
                print(f"\n--- Episode {ep+1} ---")
                print(f"Map: {raw_env.current_map_name}")

    venv.close()

if __name__ == "__main__":
    main()