import os
import sys
import json
import argparse
import numpy as np
import traceback

from robot_env import RobotNavEnv
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# ==========================================
# Helpers
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_config(experiment_folder):
    """Safely load metadata configuration from the experiment folder."""
    meta_path = os.path.join(experiment_folder, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find metadata.json at {meta_path}")
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    algo_str = meta["experiment_info"]["algorithm"]
    pf_active = meta["experiment_info"]["pf_active"]
    algo_map = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "DDPG": DDPG}
    
    if algo_str not in algo_map:
        raise ValueError(f"Unsupported algorithm found in config: {algo_str}")
        
    return algo_map[algo_str], pf_active

def make_test_env(pf_active, rank, seed=0, render=False, episodes = 100):
    """Utility to instantiate parallel environments safely."""
    def _init():
     
        env = RobotNavEnv(
            render=False, 
            pf_active=pf_active,       # Fixed: was PF_ACTIVE
            seed=seed,
            is_eval=False,   
            is_testing=True,   # Critical: set testing mode for proper map/seed handling
            worker_id=rank,            
            num_eval_episodes= episodes, # Pass total episodes for proper seed distribution
        )
        return env
    return _init

# ==========================================
# Testing Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Parallel RL Testing Script")
    parser.add_argument('--experiment-dir', type=str, default="run_SAC_True_34906", help="Name of the experiment folder inside 'models/'")
    parser.add_argument('--model-name', type=str, default="best_model", help="Name of the saved model zip/pkl (without extension)")
    parser.add_argument('--num-episodes', type=int, default=500, help="Total number of episodes to test")
    parser.add_argument('--num-envs', type=int, default=4, help="Number of parallel environments to run")
    parser.add_argument('--seed', type=int, default=12345, help="Base random seed for reproducibility")
    parser.add_argument('--render', action='store_true', help="Enable rendering (will only render worker 0 to prevent crashes)")
    args = parser.parse_args()

    NUM_EPISODES = args.num_episodes
    INITIAL_SEED = args.seed

    experiment_dir = os.path.join(MODELS_DIR, args.experiment_dir)
    model_path = os.path.join(experiment_dir, args.model_name, f"{args.model_name}.zip")
    stats_path = os.path.join(experiment_dir, args.model_name, f"{args.model_name}.pkl")

    # Fallback paths in case best_model is directly in the experiment root
    if not os.path.exists(model_path):
        model_path = os.path.join(experiment_dir, f"{args.model_name}.zip")
        stats_path = os.path.join(experiment_dir, f"{args.model_name}.pkl")

    print(f"[{args.experiment_dir}] Starting testing session...")
    print(f"Base Seed: {args.seed} | Target Episodes: {args.num_episodes} | Parallel Workers: {args.num_envs}")

    # 1. Detect Algorithm and PF status
    AlgoClass, pf_active = load_config(experiment_dir)
    
    env = None
    try:
        # 2. Create the Parallel Environments
        print("Initializing parallel environments...")
        venv = SubprocVecEnv([make_test_env(pf_active, i, args.seed, args.render, args.num_episodes) for i in range(args.num_envs)])
        
        set_random_seed(args.seed)

        # 3. Handle Normalization (CRITICAL for test accuracy)
        if os.path.exists(stats_path):
            print(f"Loading VecNormalize statistics from {stats_path}")
            env = VecNormalize.load(stats_path, venv)
            # MUST disable updating during testing
            env.training = False
            env.norm_reward = False
        else:
            print("WARNING: No VecNormalize .pkl found. Running unnormalized.")
            env = venv

        # 4. Load Model
        print(f"Loading {AlgoClass.__name__} model...")
        model = AlgoClass.load(model_path, env=env)

        # 5. Parallel Testing Loop
        print(f"\nTesting {AlgoClass.__name__} (PF={pf_active}) across {args.num_envs} workers...")
        
        obs = env.reset()
        episodes_completed = 0
        successful_episodes = 0
        collision_episodes = 0
        timeout_episodes = 0
        collision_goal_episodes = 0

        episode_rewards = []
        episode_steps = []

        while episodes_completed < args.num_episodes:
            # Deterministic=True is standard for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    
                    # Accumulate stats
                    episodes_completed += 1
                    is_success = info.get('success', False)
                    is_collision = info.get('collision', False)
                    is_timeout = info.get("TimeLimit.truncated", False)
                    is_collision_goal = info.get('goal_crash', False)
                    if is_success:
                        successful_episodes += 1
                    if is_collision:
                        collision_episodes += 1
                    if is_timeout:
                        timeout_episodes += 1
                    if is_collision_goal:
                        collision_goal_episodes += 1
                    # 'episode' dict is populated by SB3 Monitor/VecEnv when an episode finishes
                    if 'episode' in info:
                        ep_reward = info['reward']
                        ep_length = info['steps']
                        episode_rewards.append(ep_reward)
                        episode_steps.append(ep_length)
                    
                    result_str = "SUCCESS" if is_success else "COLLISION_GOAL" if is_collision_goal else "COLLISION" if is_collision else "TIMEOUT" if is_timeout else "UNKNOWN"
                    target_error = info.get('target_error', 0.0)
                    
                    print(f"Worker {i:02d} | Ep {episodes_completed:03d}/{args.num_episodes:03d} | "
                          f"Result: {result_str} | Error: {target_error:.2f}m")

                    if episodes_completed >= args.num_episodes:
                        break # Break out of the for-loop if we hit the target

        # 6. Final Statistics
        print("\n" + "="*50)
        print("TESTING COMPLETE")
        print("="*50)
        success_rate = (successful_episodes / args.num_episodes) * 100
        collision_rate = (collision_episodes / args.num_episodes) * 100
        timeout_rate = (timeout_episodes / args.num_episodes) * 100
        collision_goal_rate = (collision_goal_episodes / args.num_episodes) * 100
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_steps = np.mean(episode_steps) if episode_steps else 0.0
        
        print(f"Total Episodes : {args.num_episodes}")
        print(f"Success Rate   : {success_rate:.2f}% ({successful_episodes}/{args.num_episodes})")
        print(f"Collision Rate : {collision_rate:.2f}% ({collision_episodes}/{args.num_episodes})")
        print(f"Timeout Rate   : {timeout_rate:.2f}% ({timeout_episodes}/{args.num_episodes})")
        print(f"Collision Goal Rate : {collision_goal_rate:.2f}% ({collision_goal_episodes}/{args.num_episodes})")
        print(f"Average Reward : {mean_reward:.2f}")
        print(f"Average Steps  : {mean_steps:.1f}")

        print("="*50 + "\n")

    except Exception as e:
        print("\n[CRITICAL ERROR] Testing interrupted!")
        print(f"Exception details: {e}")
        traceback.print_exc()
        
    finally:
        # Guaranteed cleanup block matching your robust training setup
        print("[Cleanup] Shutting down environments safely...")
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()