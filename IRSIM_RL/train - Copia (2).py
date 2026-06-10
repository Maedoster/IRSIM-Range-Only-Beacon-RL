import json
import shutil

import os
import platform
import sys
import gymnasium
import psutil
from irsim import env
import numpy as np
import stable_baselines3
import torch
import yaml
import argparse
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from robot_env import RobotNavEnv
from stable_baselines3 import TD3, PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

#Da Fixare
#1) Se il robot non punta verso il goal quando A* genera il path, il robot non riesce ad arrivare al goal, non si gira verso il waypoint (anche senza turn penalty)
#2) A* ci mette parecchio ad attivarsi a generare il path anche quando la stima è parecchio accurata
#3) Anzichè resettare ambiente completamente ad ogni reset con irsim.make vedere se ci sono alternative

# ==========================================
# CONFIGURATION 
# ==========================================

SELECTED_ALGORITHM = "SAC"   # Options: "DDPG", "TD3", "PPO", "SAC"
PF_ACTIVE = True            # Set to True for Particle Filter, False for Least Squares
INITIAL_SEED = np.random.randint(0, 100000) #Or a specific one for reproducibility, e.g., 12345 
#INITIAL_SEED = 12345

# --- Training Parameters ---
NUM_ENVS = 8
TOTAL_TIMESTEPS = 1000000
TOTAL_MAPS_TO_USE = 100

# --- Resume Settings ---
# Set RESUME_FOLDER to None if starting a fresh run
RESUME_FOLDER = None  # e.g., "models/run_SAC_True_12345" or "models/run_SAC_True_12345/crash_checkpoints"
CHECKPOINT_NAME = "last_checkpoint"  # e.g., "best_model" or "SAC_recovery_380000_steps"

# ==========================================
# Directory Setup & Helpers
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MAP_DIR = os.path.join(PROJECT_ROOT, "HouseExpo_IRSim_Converter", "IRSimDataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WORLDS_DIR, exist_ok=True)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)  

def make_env(world_file, pf_active, rank, seed=0, log_file=None):
    def _init():
        env = RobotNavEnv(render=False, world_file=world_file, pf_active=pf_active, seed=seed + rank)
        # Wrap the environment in the shield with logging enabled
        env = ExceptionShieldWrapper(env, log_file=log_file, rank=rank)
        return env
    return _init

def make_eval_env(map_partition, pf_active, rank, seed=0, log_file=None):
    def _init():
        initial_map = map_partition[0] if map_partition else ""
        env = RobotNavEnv(render=False, world_file=initial_map, pf_active=pf_active, seed=seed + rank)
        
        # Add the cycling wrapper here so training envs actually cycle
        env = MapCyclingWrapper(env, map_partition)
        
        # Wrap the environment in the shield with logging enabled
        env = ExceptionShieldWrapper(env, log_file=log_file, rank=rank)
        return env
    return _init


# ==========================================
# Callbacks
# ==========================================

class ExceptionShieldWrapper(gymnasium.Wrapper):
    """
    Catches Python-level exceptions inside the environment to prevent
    the workers/main thread from dying. Forces a reset upon failure.
    """
    def __init__(self, env, log_file=None, rank=0):
        super().__init__(env)
        self.log_file = log_file
        self.rank = rank

    def _log_crash(self, method_name, exception):
        print(f"\n[Env Error - Rank {self.rank}] Caught exception in {method_name}(): {exception}")
        traceback.print_exc()
        
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"=== [ENV CRASH] {np.datetime64('now')} | Rank: {self.rank} | Method: {method_name} ===\n")
                    f.write(traceback.format_exc())
                    f.write("\n" + "="*60 + "\n\n")
            except Exception as write_err:
                print(f"[Warning] Failed to write to env crash log: {write_err}")

    def step(self, action):
        try:
            return self.env.step(action)
        except Exception as e:
            self._log_crash("step", e)
            
            # Match the 5-tuple Gymnasium standard
            dummy_obs = self.env.observation_space.sample()
            dummy_reward = -10.0  # Heavy penalty for breaking
            terminated = True
            truncated = False
            info = {"success": False, "error_caught": str(e)}
            
            return dummy_obs, dummy_reward, terminated, truncated, info

    def reset(self, **kwargs):
        try:
            return self.env.reset(**kwargs)
        except Exception as e:
            self._log_crash("reset", e)
            
            # Match the 2-tuple Gymnasium standard (obs, info)
            dummy_obs = self.env.observation_space.sample()
            info = {"success": False, "error_caught": str(e)}
            return dummy_obs, info
        
class DeterministicResetWrapper(gymnasium.Wrapper):
    """
    Forces the environment to reset using a specific, fixed seed 
    (base + rank) every single time.
    """
    def __init__(self, env, full_seed):
        super().__init__(env)
        self.fixed_seed = full_seed

    def reset(self, **kwargs):
        # Rimuoviamo eventuali seed passati dall'esterno
        kwargs.pop('seed', None)
        # Forza il seed specifico calcolato per questo rank
        return self.env.reset(seed=self.fixed_seed, **kwargs)
    
class MapCyclingWrapper(gymnasium.Wrapper):
    def __init__(self, env, map_paths):
        super().__init__(env)
        self.map_paths = map_paths
        self.current_idx = 0
        
    def reset(self, **kwargs):
        # Simply pick the map
        options = kwargs.get("options", {})
        options["current_map"] = self.map_paths[self.current_idx]
        options["is_eval"] = True
        kwargs["options"] = options
        
        # Increment for the NEXT reset
        self.current_idx = (self.current_idx + 1)
        
        return self.env.reset(**kwargs)


class TrainingProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        self.start_time = None

        self.process = psutil.Process(os.getpid())

        self.start_steps = None  # Add this tracker

    def _on_step(self) -> bool:
        if self.start_time is None:
            self.start_time = time.time()
            self.start_steps = self.num_timesteps  # Lock the starting point (e.g., 380000)

        # Estrai i dati di successo da tutti i sub-ambienti (SubprocVecEnv)
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                if "success" in info:
                    self.episode_successes.append(float(info["success"]))

        # Ogni check_freq passi (totali), calcola e stampa
        if self.n_calls % self.check_freq == 0:
            # 1. Success Rate
            if len(self.episode_successes) > 0:
                success_rate = (sum(self.episode_successes) / len(self.episode_successes)) * 100
            else:
                success_rate = 0.0

            # 2. ETA (Tempo Rimanente)
            elapsed_time = time.time() - self.start_time

            session_steps_done = self.num_timesteps - self.start_steps
            steps_per_sec = session_steps_done / elapsed_time if elapsed_time > 0 else 0
            
            steps_done = self.num_timesteps
            total_steps = self.locals.get("_total_timesteps", self.model._total_timesteps)
            
            # Calcolo velocità media
            remaining_steps = total_steps - steps_done
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            ram_bytes = self.process.memory_info().rss
            ram_mb = ram_bytes / (1024 ** 2)  # Convert to Megabytes
            
            if torch.cuda.is_available():
                # CHOOSE ONE (Or log both):
                # torch.cuda.memory_reserved() shows the total memory locked by the PyTorch caching allocator
                vram_bytes = torch.cuda.memory_reserved() 
                vram_mb = vram_bytes / (1024 ** 2)
                vram_str = f"{vram_mb:.1f} MB"
            else:
                vram_str = "N/A (CPU Mode)"

            # Log a terminale
            print(f"\n>>> [PROGRESS] Step: {steps_done}/{total_steps}")
            print(f">>> Success Rate: {success_rate:.2f}% (last {len(self.episode_successes)} ep)")
            print(f">>> Speed: {steps_per_sec:.1f} steps/s | ETA: {eta_str}")
            print(f">>> [MEMORY OVERHEAD] System RAM: {ram_mb:.1f} MB | CUDA VRAM: {vram_str}")
            
            # Log su Tensorboard
            self.logger.record("metrics/success_rate_period", success_rate)
            self.logger.record("time/steps_per_second", steps_per_sec)

            # Reset per la prossima finestra temporale
            self.episode_successes = []

        return True

class EvalAndSaveVecNormalizeCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(EvalAndSaveVecNormalizeCallback, self).__init__(*args, **kwargs)
        self.last_best_reward = -float("inf")

    def _on_step(self) -> bool:
        # Sync stats from training env to eval env before evaluating
        if self.eval_env is not None and self.training_env is not None:
            sync_envs_normalization(self.training_env, self.eval_env)

        continue_training = super()._on_step()
        
        if self.best_mean_reward > self.last_best_reward:
            self.last_best_reward = self.best_mean_reward
            if self.best_model_save_path is not None:
                stats_path = os.path.join(self.best_model_save_path, "best_model.pkl")
                
                # ---> FIX: Use the native SB3 getter to safely grab the VecNormalize env <---
                vec_env = self.model.get_vec_normalize_env() 
                
                if vec_env is not None:
                    vec_env.save(stats_path)
                else:
                    if self.verbose > 0:
                        print("\n[EvalCallback] WARNING: VecNormalize wrapper not found.")
        return continue_training
    

class CheckpointWithVecNormalizeCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        # 1. Let the parent handle the standard "numbered" saves
        result = super()._on_step() 
        
        if self.n_calls % self.save_freq == 0:
            # --- Paths ---
            last_stats_path = os.path.join(self.save_path, "last_checkpoint.pkl")
            last_buffer_path = os.path.join(self.save_path, "last_replay_buffer.pkl")
            last_model_path = os.path.join(self.save_path, "last_checkpoint.zip") # New

            history_stats_path = os.path.join(self.save_path, f"{SELECTED_ALGORITHM}_recovery_{self.num_timesteps}_steps.pkl")
            
            # --- Save VecNormalize (The "Last" version) ---
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                tmp_stats = last_stats_path + ".tmp"
                vec_env.save(tmp_stats)
                vec_env.save(history_stats_path)  # Save current stats for history
                shutil.move(tmp_stats, last_stats_path)
                
            # --- Save Replay Buffer (The "Last" version) ---
            if hasattr(self.model, "save_replay_buffer"):
                tmp_buffer = last_buffer_path + ".tmp"
                self.model.save_replay_buffer(tmp_buffer)
                shutil.move(tmp_buffer, last_buffer_path)

            # --- Save Model Weights (The "Last" version) ---
            tmp_model = last_model_path + ".tmp"
            self.model.save(tmp_model)
            shutil.move(tmp_model, last_model_path)
                
        return result

class TrainingSchedulerCallback(BaseCallback):
    def __init__(self, algorithm_name, verbose=1):
        super(TrainingSchedulerCallback, self).__init__(verbose)
        self.algorithm_name = algorithm_name
        self.decay_rate = 0.9999
        self.episode_count = 0
        self.session_steps = 0  # Track steps taken ONLY in this run snippet

    def _on_step(self) -> bool:
        self.session_steps += 1 # Increment every environment step execution
        
        if "dones" in self.locals and any(self.locals["dones"]):
            self.episode_count += 1
            if self.algorithm_name in ["TD3", "DDPG"]:
                if self.model.action_noise is not None:
                    if hasattr(self.model.action_noise, 'base_noise'):
                        self.model.action_noise.base_noise._sigma *= self.decay_rate
                    else:
                        self.model.action_noise._sigma *= self.decay_rate

            # Check pacing using session-isolated steps instead of absolute steps
            if self.session_steps % 5000 == 0:
                if self.model.batch_size < 256:
                    self.model.batch_size = min(self.model.batch_size + 2, 256)
                    if self.verbose > 0:
                        print(f"\n[Scheduler] Incremented batch_size to {self.model.batch_size}")
        return True
    
class TensorboardCustomVarsCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            errors = [info.get("target_error") for info in infos if "target_error" in info]
            if errors:
                self.logger.record("metrics/target_error", np.mean(errors))
        return True

# ==========================================
# Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=NUM_ENVS) 
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS) 
    parser.add_argument('--resume-folder', type=str, default=RESUME_FOLDER)
    parser.add_argument('--checkpoint-name', type=str, default=CHECKPOINT_NAME)
    args = parser.parse_args()

    # --- 1) GATHER AND PARTITION THE MAPS ---
    if not os.path.exists(MAP_DIR):
        print(f"Error: Directory {MAP_DIR} not found.")
        return
        
    all_map_files = sorted([f for f in os.listdir(MAP_DIR) if f.endswith('.yaml')])
    if len(all_map_files) == 0:
        print(f"Error: No yaml maps found in {MAP_DIR}.")
        return
    
    if len(all_map_files) < TOTAL_MAPS_TO_USE:
        print(f"Warning: Only {len(all_map_files)} maps found, using all of them.")
        selected_maps = all_map_files
    else:
        selected_maps = all_map_files[:TOTAL_MAPS_TO_USE]

    selected_map_paths = [os.path.join(MAP_DIR, m) for m in selected_maps]
    
    # Split the selected maps evenly across the number of environments
    map_partitions = []
    chunk_base_size, chunk_remainder = divmod(len(selected_map_paths), args.num_envs)
    
    for i in range(args.num_envs):
        start = i * chunk_base_size + min(i, chunk_remainder)
        end = (i + 1) * chunk_base_size + min(i + 1, chunk_remainder)
        map_partitions.append(selected_map_paths[start:end])
        print(f"Env {i} assigned {len(map_partitions[i])} maps (e.g. {os.path.basename(map_partitions[i][0])} to {os.path.basename(map_partitions[i][-1])})")
    # -----------------------------------------

    map_filename = "0004d52d1aeeb8ae6de39d6bd993e992.yaml" 
    map_path = os.path.join(MAP_DIR, map_filename)
    if not os.path.exists(map_path):
        print(f"Error: Map file {map_path} not found.")
        return
    
    # Determine Experiment Folder logic
    if args.resume_folder:
        if os.path.basename(args.resume_folder) == "crash_checkpoints":
            experiment_folder = os.path.dirname(args.resume_folder)
        else:
            experiment_folder = args.resume_folder
        
        print(f"\n[Resume] Resuming training. Main folder: {experiment_folder}")

        orig_meta_path = os.path.join(experiment_folder, "metadata.json")
        if os.path.exists(orig_meta_path):
            with open(orig_meta_path, "r") as f:
                old_meta = json.load(f)
                # Extract the seed safely from your JSON structure
                initial_seed = int(old_meta["experiment_info"]["initial_seed"])
                model_suffix = f"{SELECTED_ALGORITHM}_{PF_ACTIVE}_{initial_seed}"
            print(f"[Resume] Found original seed in metadata! Lock-in seed: {initial_seed}")
        else:
            initial_seed = INITIAL_SEED  # Fallback to global constant if metadata is missing
            print(f"[Resume] WARNING: metadata.json not found in {experiment_folder}. Using current INITIAL_SEED.")
        
    else:
        initial_seed = INITIAL_SEED  # Use the global constant for fresh runs
        model_suffix = f"{SELECTED_ALGORITHM}_{PF_ACTIVE}_{initial_seed}"
        experiment_folder = os.path.join(MODELS_DIR, f"run_{model_suffix}")
        os.makedirs(experiment_folder, exist_ok=True)

    
    # Dynamic variables tracking recovery states across loops
    current_resume_folder = args.resume_folder
    current_checkpoint_name = args.checkpoint_name
    
    max_recovery_attempts = 200
    recovery_count = 0

    # ==========================================
    # AUTO-RECOVERY TRAINING LOOP
    # ==========================================
    while recovery_count < max_recovery_attempts:
        env = None
        eval_env = None
        
        try:

            eval_seed = 99999  
            num_eval_envs = NUM_ENVS # Number of parallel evaluation processes
            env_crash_log = os.path.join(experiment_folder, "robot_env_crashes.txt")
            
            # --- Evaluation setup using SubprocVecEnv ---
            # Create a list of initialization functions
            eval_env_fns = [
                make_eval_env(map_partitions[i], PF_ACTIVE, eval_seed, i, env_crash_log) 
                for i in range(num_eval_envs)
            ]
            
            # Initialize with SubprocVecEnv
            eval_env = SubprocVecEnv(eval_env_fns)
            
            # Create the evaluation wrapper
            eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
            eval_env.seed(eval_seed)
            # ---------------------------------------------

            best_model_path = os.path.join(experiment_folder, f"best_model")
                
            eval_callback = EvalAndSaveVecNormalizeCallback(
                eval_env, 
                best_model_save_path=best_model_path,
                log_path=os.path.join(experiment_folder, "logs"),
                eval_freq=max(1000 // args.num_envs, 1), 
                n_eval_episodes=16,
                deterministic=True,
                verbose=1
            )
            
            checkpoint_callback = CheckpointWithVecNormalizeCallback(
                save_freq=max(20000 // args.num_envs, 1), 
                save_path=os.path.join(experiment_folder, "crash_checkpoints"),
                name_prefix=f"{SELECTED_ALGORITHM}_recovery"
            )

            callback_list = CallbackList([
                eval_callback, 
                checkpoint_callback, 
                TrainingSchedulerCallback(SELECTED_ALGORITHM), 
                TensorboardCustomVarsCallback(),
                TrainingProgressCallback(check_freq=max(5000 // args.num_envs, 1)),
            ])

            # TRAINING ENV
            print(f"Initializing {args.num_envs} environments for {SELECTED_ALGORITHM} (PF: {PF_ACTIVE})...")
            env = SubprocVecEnv([make_env(map_path, PF_ACTIVE, i, INITIAL_SEED, env_crash_log) for i in range(args.num_envs)])
            
            if current_resume_folder:
                stats_path = os.path.join(experiment_folder, f"{current_checkpoint_name}.pkl")
                if not os.path.exists(stats_path):
                    stats_path = os.path.join(experiment_folder, "crash_checkpoints", f"{current_checkpoint_name}.pkl")
                env = VecNormalize.load(stats_path, env)
                env.training = True
                env.norm_reward = False
            else:
                env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False, clip_obs=10.0)

            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
            
            algo_classes = {"TD3": TD3, "PPO": PPO, "DDPG": DDPG, "SAC": SAC}
            AlgoClass = algo_classes.get(SELECTED_ALGORITHM)

            if current_resume_folder:
                model_path = os.path.join(experiment_folder, current_checkpoint_name)
                if not os.path.exists(model_path + ".zip"):
                    model_path = os.path.join(experiment_folder, "crash_checkpoints", args.checkpoint_name)
                model = AlgoClass.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

                if SELECTED_ALGORITHM in ["SAC", "TD3", "DDPG"]:
                    buffer_path = os.path.join(experiment_folder, "last_replay_buffer.pkl")
                    if not os.path.exists(buffer_path):
                        buffer_path = os.path.join(experiment_folder, "crash_checkpoints", "last_replay_buffer.pkl")
                    
                    if os.path.exists(buffer_path):
                        print(f"[Resume] Loading replay buffer from {buffer_path}...")
                        model.load_replay_buffer(buffer_path)
                    else:
                        print("\n[Warning] No replay buffer found! Agent will start with empty memory.")

            else:
                # FRESH MODEL
                if SELECTED_ALGORITHM == "TD3":
                    model = TD3("MlpPolicy", env, verbose=1, seed=initial_seed, device="cuda", learning_rate=1e-3, buffer_size=500000,
                                batch_size=256, gamma=0.99, tau=0.01, train_freq=(64, "step"), gradient_steps=64,
                                learning_starts=10000, policy_delay=2, action_noise=action_noise,
                                policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
                elif SELECTED_ALGORITHM == "PPO":
                    model = PPO("MlpPolicy", env, verbose=1, seed=initial_seed, device="cpu", learning_rate=3e-4, n_steps=2048,
                                batch_size=256, n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
                                policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
                elif SELECTED_ALGORITHM == "DDPG":
                    model = DDPG("MlpPolicy", env, verbose=1, seed=initial_seed, device="cuda", learning_rate=1e-3, buffer_size=500000,
                                batch_size=256, gamma=0.99, tau=0.01, train_freq=(64, "step"), gradient_steps=64,
                                learning_starts=10000, action_noise=action_noise, policy_kwargs=policy_kwargs,
                                tensorboard_log="./logs/")
                elif SELECTED_ALGORITHM == "SAC":
                    model = SAC("MlpPolicy", env, verbose=1, seed=initial_seed, device="cuda", learning_rate=1e-3, buffer_size=500000,
                                batch_size=256, gamma=0.99, tau=0.01, train_freq=(64, "step"), gradient_steps=64,
                                learning_starts=10000, ent_coef="auto:0.005", target_entropy="auto",
                                policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
                else:
                    print(f"Unsupported algorithm: {SELECTED_ALGORITHM}")
                    return
            
            # ==========================================
            # PRE-TRAINING METADATA & REPRODUCIBILITY
            # ==========================================
            world_data = load_yaml(map_path)
            robot_data = world_data['robot'][0]
            sensor_data = robot_data['sensors'][0]

            robot_env_stats = eval_env.env_method("get_env_stats")[0]
            reward_params = eval_env.env_method("get_reward_config")[0]
            
            common_params = [
                "learning_rate", "gamma", "batch_size", "verbose", "seed", 
                "device", "buffer_size", "learning_starts", "tau", "train_freq", 
                "gradient_steps", "n_steps", "n_epochs", "ent_coef", "gae_lambda", 
                "clip_range", "target_entropy", "policy_delay"
            ]
            extracted_params = {p: getattr(model, p, None) for p in common_params if hasattr(model, p)}

            env_info = {
                "map_file": "Full Dataset from HouseExpo_IRSim_Converter/IRSimDataset",
                "max_steps": eval_env.get_attr("max_steps")[0],
            }

            robot_env_settings = {
                "physical_limits": {
                    "radius": robot_data['shape'].get('radius'),
                    "goal_threshold": robot_data.get('goal_threshold'),
                    "max_vel_yaml": robot_data.get('vel_max'),
                    "kinematics": robot_data['kinematics'].get('name')
                },
                "sensor_lidar": {
                    "range_max": sensor_data.get('range_max'),
                    "beams": sensor_data.get('number'),
                    "noise_std": sensor_data.get('std'),
                    "offset": sensor_data.get('offset') 
                },
            }

            metadata = {
                "experiment_info": {
                    "algorithm": SELECTED_ALGORITHM,
                    "pf_active": PF_ACTIVE,
                    "initial_seed": initial_seed,
                    "total_timesteps": args.total_timesteps,
                    "num_envs": args.num_envs,
                    "status": "started" if not current_resume_folder else "resumed",
                    "restarts": recovery_count,
                    "start_time": str(np.datetime64('now')),
                    "resumed_from": current_checkpoint_name if current_resume_folder else None
                },
                "system_info": {
                    "sb3_version": stable_baselines3.__version__,
                    "torch_version": torch.__version__,
                    "device": str(model.device),
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "python_version": platform.python_version(),
                },
                "environment_config": env_info,
                "robot_env_stats": robot_env_stats,
                "hyperparameters": extracted_params,
                "robot_goal_settings": robot_env_settings, 
                "reward_structure": reward_params,
                "vec_normalize_settings": {
                    "norm_obs": env.norm_obs,
                    "norm_reward": env.norm_reward,
                    "clip_obs": env.clip_obs,
                    "initial_obs_rms_mean": float(np.mean(env.obs_rms.mean)) if env.obs_rms else None
                }
            }

            # Save Metadata (Appending a suffix if it's a resumed run to avoid overwriting original data)
            meta_name = "metadata_resumed.json" if current_resume_folder else "metadata.json"
            with open(os.path.join(experiment_folder, meta_name), "w") as f:
                json.dump(metadata, f, indent=4, default=str)

            print(f"\n[Metadata] Configuration saved to {os.path.join(experiment_folder, meta_name)}")
            print(f"[Training] Starting {args.total_timesteps} timesteps...")

            # Set reset_num_timesteps based on whether we are resuming or not
            do_reset_timesteps = False if current_resume_folder else True
            model.learn(total_timesteps=args.total_timesteps, callback=callback_list, reset_num_timesteps=do_reset_timesteps)

            # Save Final Model
            model.save(os.path.join(experiment_folder, f"model_final"))
            vec_env = model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(os.path.join(experiment_folder, f"model_final.pkl"))

            print(f"\nTraining Complete. Experiment folder: {experiment_folder}")


        except Exception as e:
            recovery_count += 1
            print(f"\n{'!'*60}\n[CRASH INTERCEPTED] Training loop halted on recovery attempt {recovery_count}/{max_recovery_attempts}!")
            print(f"Exception details: {e}")
            traceback.print_exc()
            print(f"{'!'*60}\n")
            
            # Log crash details explicitly to file
            crash_log_file = os.path.join(experiment_folder, "irsim_auto_recovery_log.txt")
            with open(crash_log_file, "a") as log:
                log.write(f"=== CRASH TIMESTAMP: {np.datetime64('now')} (Attempt #{recovery_count}) ===\n")
                traceback.print_exc(file=log)
                log.write("\n" + "="*50 + "\n\n")
            
            # Clean up active process vectors explicitly to prevent socket/shared memory deadlocks
            try:
                if env is not None:
                    print("[Auto-Recovery] Terminating stale training subprocess workers...")
                    env.close()
                if eval_env is not None:
                    print("[Auto-Recovery] Terminating stale validation environment context...")
                    eval_env.close()
            except Exception as cleanup_error:
                print(f"[Auto-Recovery Warning] Minor error encountered during pipeline cleanup: {cleanup_error}")
            
            # Check if a last_checkpoint actually exists to load from
            backup_chk_path = os.path.join(experiment_folder, "crash_checkpoints", "last_checkpoint.zip")
            if os.path.exists(backup_chk_path):
                print(f"[Auto-Recovery] Found last safe snapshot file: {backup_chk_path}")
                print("[Auto-Recovery] Overriding runtime parameters to perform checkpoint rollback recovery on next loop pass...")
                current_resume_folder = experiment_folder
                current_checkpoint_name = "last_checkpoint"
            else:
                print("\n[CRITICAL ERROR] No 'last_checkpoint.zip' exists.")
                print("[Action] No checkpoint to recover from. Shutting down to prevent infinite loop.")
                sys.exit(1)
                    
            print("[Auto-Recovery] Sleeping for 5 seconds to cool down hardware parameters before execution rebirth...")
            time.sleep(5)
            print("[Auto-Recovery] Initiating fresh simulation session block structure now...\n")

if __name__ == '__main__':
    main()