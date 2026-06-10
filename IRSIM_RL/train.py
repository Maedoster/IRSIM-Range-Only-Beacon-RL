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
#2) Far arrivare il robot al goal con un buon rateo senza collisioni 
#3) Implementare RTTP*
#4) Implementare dummyVecEnv per evaluation come scelta possibile


# ==========================================
# CONFIGURATION 
# ==========================================

SELECTED_ALGORITHM = "SAC"   # Options: "DDPG", "TD3", "PPO", "SAC"
PF_ACTIVE = True            # Set to True for Particle Filter, False for Least Squares
INITIAL_SEED = np.random.randint(0, 100000) #Or a specific one for reproducibility, e.g., 12345 
#INITIAL_SEED = 12345

# --- Training Parameters ---
NUM_ENVS = 8

USE_DUMMY_EVAL = False  # Set to True to use DummyVecEnv for evaluation (single environment, no parallelism)
NUM_EVAL_ENVS = 8  # Only relevant if USE_DUMMY_EVAL is False. Number of parallel environments for evaluation.

TOTAL_TIMESTEPS = 3000000
EVAL_EPISODES = 100

SAVE_FREQ = 20000  # Save every N environment steps (adjusted by number of envs in callbacks)
EVAL_FREQ = 30000   # Evaluate every N environment steps (adjusted by number of envs in callbacks)

# --- Resume Settings ---
# Set RESUME_FOLDER to None if starting a fresh run
# Use a string for Windows path. Set to None to start fresh.
RESUME_FOLDER = None  # r"C:\Users\tomma\Desktop\Tesi Magistrale\Progetto\IRSIM_RL\models\run_SAC_True_78942\crash_checkpoints"  # e.g., "models/run_SAC_True_12345" or "models/run_SAC_True_12345/crash_checkpoints"
CHECKPOINT_NAME = "last_checkpoint"  # e.g., "best_model" or "SAC_recovery_380000_steps"


# ==========================================
# GLOBAL VARIABLES
# ==========================================

eval_env = None  # <--- Define eval_env globally


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

def make_env(world_file, pf_active, rank, seed=0, log_file=None, num_workers=1):
    def _init():
        env = RobotNavEnv(
            render=False, 
            world_file=world_file, 
            pf_active=pf_active, 
            seed=seed + rank,
            is_eval=False,            # <--- Explicitly set to False for training
            worker_id=rank,           # <--- Pass rank
            num_workers=num_workers,   # <--- Pass total workers
            num_eval_episodes= EVAL_EPISODES       # <--- Not used in training, but set to 0 for clarity
        )
        # Wrap the environment in the shield with logging enabled
        env = ExceptionShieldWrapper(env, log_file=log_file, rank=rank)
        return env
    return _init

# ==========================================
# EVALUATION SETUP
# ==========================================
def make_eval_env(world_file, pf_active, rank, seed=0, log_file=None, num_workers=1):
    def _init():
        raw_env = RobotNavEnv(
            render=False, 
            world_file=world_file,     # Fixed: was map_path
            pf_active=pf_active,       # Fixed: was PF_ACTIVE
            seed=seed,
            is_eval=True,   
            is_serial_eval=USE_DUMMY_EVAL,           
            worker_id=rank,            
            num_workers=num_workers,    # Fixed: was NUM_EVAL_ENVS
            num_eval_episodes=EVAL_EPISODES  
            
        )
                    
        # Ensure deterministic resetting across evaluations
        shielded_eval_env = ExceptionShieldWrapper(raw_env, log_file=log_file, rank=rank) # Fixed: was env_crash_log
                    
        # Important: Monitor must be applied *before* vectorization
        return Monitor(shielded_eval_env)
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
            dummy_reward = 0.0  # Heavy penalty for breaking
            terminated = False
            truncated = True
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



class TrainingProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, experiment_folder: str = None, verbose=1):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.experiment_folder = experiment_folder
        self.start_time = None
        self.process = psutil.Process(os.getpid())
        self.start_steps = None
        
        # Performance Tracking Lists
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_truncations = []
        
        # Memory tracking flags and baselines
        self.saved_5000_stats = False
        self.baseline_ram_mb = None
        self.baseline_vram_mb = None
        self.has_recorded_5000 = False  # New flag to ensure we only record once

        # --- Restore from JSON if resuming ---
        if self.experiment_folder:
            meta_default_path = os.path.join(self.experiment_folder, "metadata.json")
            
            if os.path.exists(meta_default_path):
                try:
                    with open(meta_default_path, 'r') as f:
                        meta_data = json.load(f)
                    
                    if "periodic_tracking" in meta_data and "first_5000_steps" in meta_data["periodic_tracking"]:
                        self.saved_5000_stats = True
                        self.has_recorded_5000 = True  # Mark as already recorded
                        self.baseline_ram_mb = meta_data["periodic_tracking"]["first_5000_steps"].get("ram_mb")
                        self.baseline_vram_mb = meta_data["periodic_tracking"]["first_5000_steps"].get("vram_mb")
                        if self.verbose > 0:
                            print(f"[Progress] Resumed memory baselines from metadata.")
                except Exception as e:
                    print(f"[Warning] Could not load baselines from metadata: {e}")

    def _update_periodic_metadata(self, key_name: str, ram_mb: float, vram_mb: float, step: int, success_rate: float, collision_rate: float, truncated_rate: float):
        """Helper to inject memory AND performance stats into the metadata JSON safely."""
        if not self.experiment_folder:
            return
            
        meta_default_path = os.path.join(self.experiment_folder, "metadata.json")
        
        # Use a lock file to prevent concurrent writes (for multiprocessing safety)
        lock_file = meta_default_path + ".lock"
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Try to acquire lock
                if os.path.exists(lock_file):
                    time.sleep(0.1)
                    continue
                    
                # Create lock file
                with open(lock_file, 'w') as lf:
                    lf.write(str(os.getpid()))
                
                # Read existing metadata
                meta_data = {}
                if os.path.exists(meta_default_path):
                    with open(meta_default_path, 'r') as f:
                        try:
                            meta_data = json.load(f)
                        except json.JSONDecodeError:
                            meta_data = {}

                # Ensure the parent dict exists
                if "periodic_tracking" not in meta_data:
                    meta_data["periodic_tracking"] = {}

                entry_data = {
                    "step": step,
                    "success_rate(%)": round(success_rate, 2),
                    "collision_rate(%)": round(collision_rate, 2),
                    "truncated_rate(%)": round(truncated_rate, 2),
                    "ram_mb": round(ram_mb, 2),
                    "vram_mb": round(vram_mb, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # For 'first_5000_steps', only write if it doesn't already exist
                if key_name == "first_5000_steps":
                    if key_name not in meta_data["periodic_tracking"]:
                        meta_data["periodic_tracking"][key_name] = entry_data
          
                elif key_name == "latest":
                    # Always update latest and history
                    meta_data["periodic_tracking"]["latest"] = entry_data


                # Write atomically with temp file
                tmp_meta = meta_default_path + ".tmp"
                with open(tmp_meta, 'w') as f:
                    json.dump(meta_data, f, indent=4)
                shutil.move(tmp_meta, meta_default_path)
                
                # Remove lock file
                os.remove(lock_file)
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"[Warning] Attempt {attempt+1} failed to write periodic stats to metadata: {e}")
                # Clean up lock file if it exists and we created it
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                    except:
                        pass
                if attempt == max_attempts - 1:
                    print(f"[Error] Failed to write metadata after {max_attempts} attempts")
                time.sleep(0.1)

    def _on_step(self) -> bool:
        if self.start_time is None:
            self.start_time = time.time()
            self.start_steps = self.num_timesteps

        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is not None and dones is not None:
            for i, done in enumerate(dones):
                if done: 
                    info = infos[i]
                    actual_info = info.get("final_info", info)
                    
                    if "success" in actual_info:
                        self.episode_successes.append(float(actual_info["success"]))
                    
                    self.episode_collisions.append(float(actual_info.get("collision", False)))
                    is_truncated = info.get("TimeLimit.truncated", False) or actual_info.get("truncated", False)
                    self.episode_truncations.append(float(is_truncated))

        if self.n_calls % self.check_freq == 0:
            total_eps = len(self.episode_successes)
            if total_eps > 0:
                success_rate = (sum(self.episode_successes) / total_eps) * 100
                collision_rate = (sum(self.episode_collisions) / total_eps) * 100
                truncated_rate = (sum(self.episode_truncations) / total_eps) * 100
            else:
                success_rate = collision_rate = truncated_rate = 0.0
            
            elapsed_time = time.time() - self.start_time
            session_steps_done = self.num_timesteps - self.start_steps
            steps_per_sec = session_steps_done / elapsed_time if elapsed_time > 0 else 0
            
            steps_done = self.num_timesteps
            total_steps = self.locals.get("_total_timesteps", self.model._total_timesteps)
            
            remaining_steps = total_steps - steps_done
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            total_ram_bytes = self.process.memory_info().rss
            try:
                for child in self.process.children(recursive=True):
                    total_ram_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            ram_mb = total_ram_bytes / (1024 ** 2)
            
            vram_allocated_mb = 0.0
            if torch.cuda.is_available():
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                vram_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)

            # --- FIXED: Only record 5000 stats ONCE and never again ---
            if not self.has_recorded_5000 and 4900 <= steps_done <= 5100:  # Narrow window
                self.baseline_ram_mb = ram_mb
                self.baseline_vram_mb = vram_allocated_mb
                self._update_periodic_metadata("first_5000_steps", ram_mb, vram_allocated_mb, steps_done, success_rate, collision_rate, truncated_rate)
                self.has_recorded_5000 = True  # Mark as recorded, will never happen again
                self.saved_5000_stats = True
                if self.verbose > 0:
                    print(f"\n[Progress] Recorded baseline stats at {steps_done} steps")
                
            # Always update latest stats (this overwrites only the 'latest' entry, not 'first_5000_steps')
            self._update_periodic_metadata("latest", ram_mb, vram_allocated_mb, steps_done, success_rate, collision_rate, truncated_rate)

            ram_delta_str = ""
            vram_delta_str = ""
            if self.saved_5000_stats and self.baseline_ram_mb is not None:
                ram_delta = ram_mb - self.baseline_ram_mb
                ram_delta_str = f" [Δ {ram_delta:+.1f} MB]"
                if torch.cuda.is_available() and self.baseline_vram_mb is not None:
                    vram_delta = vram_allocated_mb - self.baseline_vram_mb
                    vram_delta_str = f" [Δ {vram_delta:+.1f} MB]"

            if torch.cuda.is_available():
                vram_str = f"Allocated: {vram_allocated_mb:.1f} MB{vram_delta_str} / Reserved: {vram_reserved:.1f} MB"
            else:
                vram_str = "N/A (CPU Mode)"

            print(f"\n>>> [PROGRESS] Step: {steps_done}/{total_steps}")
            print(f">>> Rates: Success {success_rate:.1f}% | Collisions {collision_rate:.1f}% | Truncated {truncated_rate:.1f}% ({total_eps} eps)")
            print(f">>> Speed: {steps_per_sec:.1f} steps/s | ETA: {eta_str}")
            print(f">>> [MEMORY OVERHEAD] System RAM: {ram_mb:.1f} MB{ram_delta_str} | CUDA VRAM -> {vram_str}")
            
            self.logger.record("metrics/success_rate_period", success_rate)
            self.logger.record("metrics/collision_rate_period", collision_rate)
            self.logger.record("metrics/truncated_rate_period", truncated_rate)
            self.logger.record("time/steps_per_second", steps_per_sec)
            self.logger.record("system/ram_usage_mb", ram_mb)

            self.episode_successes = []
            self.episode_collisions = []
            self.episode_truncations = []

        return True

class EvalAndSaveBestSuccessCallback(BaseCallback):
    def __init__(self, eval_env, best_model_save_path=None, log_path=None, 
                 eval_freq: int = 10000, n_eval_episodes: int = 100, 
                 deterministic: bool = True, verbose: int = 1, experiment_folder: str = None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.experiment_folder = experiment_folder
        
        self.best_success_rate = -1.0
        self.best_mean_reward = -float("inf")

        if self.experiment_folder:
            meta_default_path = os.path.join(self.experiment_folder, "metadata.json")
            
            if os.path.exists(meta_default_path):
                try:
                    with open(meta_default_path, 'r') as f:
                        meta_data = json.load(f)
                    if "best_model_stats" in meta_data:
                        self.best_success_rate = float(meta_data["best_model_stats"].get("best_success_rate(%)", -1.0))
                        self.best_mean_reward = float(meta_data["best_model_stats"].get("best_mean_reward", -float("inf")))
                        
                        if self.verbose > 0:
                            print(f"[EVAL] Resumed best stats from metadata: {self.best_success_rate}% success, {self.best_mean_reward:.2f} reward.")
                except Exception as e:
                    print(f"[Warning] Could not load best stats from metadata: {e}")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n[EVAL] Triggered at {self.num_timesteps} timesteps (callback calls: {self.n_calls})")
            
            if self.training_env is not None:
                sync_envs_normalization(self.training_env, self.eval_env)
                
            episode_rewards = []
            episode_successes = []
            episode_collisions = []
            episode_truncations = []
            
            obs = self.eval_env.reset()
            current_rewards = np.zeros(self.eval_env.num_envs)
            episodes_completed = 0
            
            while episodes_completed < self.n_eval_episodes:
                actions, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(actions)
                current_rewards += rewards
                
                for i, done in enumerate(dones):
                    if done:
                        episode_rewards.append(current_rewards[i])
                        current_rewards[i] = 0.0
                        episodes_completed += 1
                        
                        info = infos[i]
                        actual_info = info.get("final_info", info)
                        
                        if "success" in actual_info:
                            episode_successes.append(float(actual_info["success"]))
                            
                        episode_collisions.append(float(actual_info.get("collision", False)))
                        is_truncated = info.get("TimeLimit.truncated", False) or actual_info.get("truncated", False)
                        episode_truncations.append(float(is_truncated))
                            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            if len(episode_successes) > 0:
                success_rate = (sum(episode_successes) / len(episode_successes)) * 100.0
                collision_rate = (sum(episode_collisions) / len(episode_collisions)) * 100.0
                truncated_rate = (sum(episode_truncations) / len(episode_truncations)) * 100.0
            else:
                success_rate = collision_rate = truncated_rate = 0.0
                
            if self.verbose > 0:
                print(f"[EVAL] Evaluated {episodes_completed} episodes.")
                print(f"[EVAL] Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"[EVAL] Rates -> Success: {success_rate:.1f}% | Collisions: {collision_rate:.1f}% | Truncated: {truncated_rate:.1f}%")
                
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/collision_rate", collision_rate)
            self.logger.record("eval/truncated_rate", truncated_rate)
            
            is_new_best = False
            
            if success_rate > self.best_success_rate:
                is_new_best = True
                if self.verbose > 0:
                    print(f"[EVAL] *** NEW BEST SUCCESS RATE! ({self.best_success_rate:.2f}% -> {success_rate:.2f}%) ***")
            elif success_rate == self.best_success_rate:
                if mean_reward > self.best_mean_reward:
                    is_new_best = True
                    if self.verbose > 0:
                        print(f"[EVAL] *** SAME SUCCESS RATE ({success_rate:.2f}%), BUT BETTER REWARD! ({self.best_mean_reward:.2f} -> {mean_reward:.2f}) ***")
                elif mean_reward == self.best_mean_reward:
                    if self.collision_rate < collision_rate:
                        is_new_best = True
                        if self.verbose > 0:
                            print(f"[EVAL] *** SAME SUCCESS RATE AND REWARD, BUT BETTER COLLISION RATE! ({collision_rate:.2f}% -> {self.collision_rate:.2f}%) ***")
                        
            if is_new_best:
                self.best_success_rate = success_rate
                self.best_mean_reward = mean_reward
                
                if self.best_model_save_path is not None:
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    model_path = os.path.join(self.best_model_save_path, "best_model.zip")
                    self.model.save(model_path)
                    
                    vec_env = self.model.get_vec_normalize_env()
                    if vec_env is not None:
                        stats_path = os.path.join(self.best_model_save_path, "best_model.pkl")
                        vec_env.save(stats_path)

            # ==========================================================
            # FIX: Moved metadata saving OUTSIDE the `if is_new_best` block 
            # so that EVERY evaluation is logged to history.
            # ==========================================================
            if self.experiment_folder:
                meta_default_path = os.path.join(self.experiment_folder, "metadata.json")
                meta_data = {}
                if os.path.exists(meta_default_path):
                    try:
                        with open(meta_default_path, 'r') as f:
                            meta_data = json.load(f)
                    except Exception as e:
                        print(f"[Warning] Metadata read failed. Skipping best model write to prevent data wipe: {e}")
                        return True 
                
                
                # 2. Update the best_model_stats specifically if it broke the record
                if is_new_best:
                    if "best_model_stats" not in meta_data:
                        meta_data["best_model_stats"] = {}
                        
                    meta_data["best_model_stats"]["best_success_rate(%)"] = self.best_success_rate 
                    meta_data["best_model_stats"]["associated_collision_rate(%)"] = round(collision_rate, 2)
                    meta_data["best_model_stats"]["associated_truncated_rate(%)"] = round(truncated_rate, 2)
                    meta_data["best_model_stats"]["best_mean_reward"] = float(self.best_mean_reward)
                    meta_data["best_model_stats"]["timesteps_achieved"] = self.num_timesteps
                
                try:
                    tmp_meta = meta_default_path + ".tmp"
                    with open(tmp_meta, 'w') as f:
                        json.dump(meta_data, f, indent=4)
                    shutil.move(tmp_meta, meta_default_path)
                    if self.verbose > 0:
                        print(f"[EVAL] Metadata updated with evaluation stats.")
                except Exception as e:
                    print(f"[Warning] Failed to write evaluation stats to metadata: {e}")

        return True
    

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
    global eval_env
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=NUM_ENVS) 
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS) 
    parser.add_argument('--resume-folder', type=str, default=RESUME_FOLDER)
    parser.add_argument('--checkpoint-name', type=str, default=CHECKPOINT_NAME)
    args = parser.parse_args()

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
    
    max_recovery_attempts = 20
    recovery_count = -1

    # ==========================================
    # AUTO-RECOVERY TRAINING LOOP
    # ==========================================
    while recovery_count < max_recovery_attempts:
        env = None
        eval_env = None
        
        try:

            eval_seed = 99999  # Static constant for perfect cross-run comparison

            
            env_crash_log = os.path.join(experiment_folder, "robot_env_crashes.txt")

           
            print(f"Initializing {NUM_EVAL_ENVS} parallel evaluation environments...")
            # Use a list comprehension to generate the environment initializers

            if USE_DUMMY_EVAL:
                print(">>> Using DummyVecEnv for Evaluation (Sequential on Main Thread)")
                eval_env = DummyVecEnv([make_eval_env(map_path, PF_ACTIVE, 0, eval_seed, env_crash_log, NUM_EVAL_ENVS)])
            else:
                print(">>> Using SubprocVecEnv for Evaluation (Parallel Processes)")
                eval_env = SubprocVecEnv([make_eval_env(map_path, PF_ACTIVE, i, eval_seed, env_crash_log, NUM_EVAL_ENVS) 
                for i in range(NUM_EVAL_ENVS)])
            
            set_random_seed(initial_seed)

            # Create the evaluation wrapper WITHOUT manually forcing obs_rms copies
            eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

            # Set the global seeds for torch/numpy
            set_random_seed(initial_seed)
            

            best_model_path = os.path.join(experiment_folder, f"best_model")
                
            eval_callback = EvalAndSaveBestSuccessCallback(
                eval_env, 
                best_model_save_path=best_model_path,
                log_path=os.path.join(experiment_folder, "logs"),
                eval_freq=max(EVAL_FREQ // args.num_envs, 1), 
                n_eval_episodes=EVAL_EPISODES,
                deterministic=True,
                verbose=1,
                experiment_folder=experiment_folder
            )
            
            checkpoint_callback = CheckpointWithVecNormalizeCallback(
                save_freq=max(SAVE_FREQ // args.num_envs, 1), 
                save_path=os.path.join(experiment_folder, "crash_checkpoints"),
                name_prefix=f"{SELECTED_ALGORITHM}_recovery"
            )

            callback_list = CallbackList([
                eval_callback, 
                checkpoint_callback, 
                TrainingSchedulerCallback(SELECTED_ALGORITHM), 
                TensorboardCustomVarsCallback(),
                TrainingProgressCallback(check_freq=max(5000 // args.num_envs, 1), experiment_folder=experiment_folder),
            ])
            
            # ==========================================
            # TIMESTEP EXTRACTION & SEED CALCULATION
            # ==========================================
            timesteps_completed = 0
            if current_resume_folder is not None:
                algo_classes = {"TD3": TD3, "PPO": PPO, "DDPG": DDPG, "SAC": SAC}
                AlgoClass = algo_classes.get(SELECTED_ALGORITHM)
                if recovery_count == -1:
                    recovery_count = 0
                # Locate the exact zip file
                model_path = os.path.join(experiment_folder, current_checkpoint_name)
                if not os.path.exists(model_path + ".zip"):
                    model_path = os.path.join(experiment_folder, "crash_checkpoints", current_checkpoint_name)
                
                # Headless load: We load the model strictly to peek at its internal step counter
                print(f"[Resume] Peeking into {current_checkpoint_name}.zip to extract timestep offset...")
                temp_model = AlgoClass.load(model_path, env=None, device="cpu") 
                timesteps_completed = temp_model.num_timesteps
                del temp_model  # Delete to immediately free up system RAM/VRAM
            
            # The exact, deterministic offset math
            current_seed = initial_seed + timesteps_completed + recovery_count
            
            print(f"\n[RNG] Base Seed: {initial_seed} | Timesteps Offset: {timesteps_completed} | Recovery Offset: {recovery_count}")
            print(f"[RNG] Final injected seed for this run: {current_seed}\n")
            
            # TRAINING ENV
            print(f"Initializing {args.num_envs} environments for {SELECTED_ALGORITHM} (PF: {PF_ACTIVE})...")
            
            # ---> FIX applied here: Pass current_seed to the environment!
            env = SubprocVecEnv([make_env(map_path, PF_ACTIVE, i, current_seed, env_crash_log, args.num_envs) for i in range(args.num_envs)])
            
            if current_resume_folder:
                stats_path = os.path.join(experiment_folder, f"{current_checkpoint_name}.pkl")
                if not os.path.exists(stats_path):
                    stats_path = os.path.join(experiment_folder, "crash_checkpoints", f"{current_checkpoint_name}.pkl")
                env = VecNormalize.load(stats_path, env)
                env.training = True
                env.norm_reward = False
            else:
                env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)

            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
            policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
            
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
                    model = TD3("MlpPolicy", env, verbose=1, seed=initial_seed, device="cuda", learning_rate=3e-4, buffer_size=300000,
                                batch_size=256, gamma=0.99, tau=0.01, train_freq=1, gradient_steps=1,
                                learning_starts=10000, policy_delay=2, action_noise=action_noise,
                                policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
                elif SELECTED_ALGORITHM == "PPO":
                    model = PPO("MlpPolicy", env, verbose=1, seed=initial_seed, device="cpu", learning_rate=3e-4, n_steps=2048,
                                batch_size=256, n_epochs=10, ent_coef=0.01, clip_range=0.2, gae_lambda=0.95,
                                policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
                elif SELECTED_ALGORITHM == "DDPG":
                    model = DDPG("MlpPolicy", env, verbose=1, seed=initial_seed, device="cuda", learning_rate=3e-4, buffer_size=300000,
                                batch_size=256, gamma=0.99, tau=0.01, train_freq=1, gradient_steps=1,
                                learning_starts=10000, action_noise=action_noise, policy_kwargs=policy_kwargs,
                                tensorboard_log="./logs/")
                elif SELECTED_ALGORITHM == "SAC":
                    model = SAC("MlpPolicy", env, verbose=1, seed=initial_seed, device="cuda", learning_rate=3e-4, buffer_size=300000,
                                batch_size=256, gamma=0.99, tau=0.01, train_freq=1, gradient_steps=1,
                                learning_starts=10000, ent_coef="auto", target_entropy="auto",
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

            robot_env_stats = env.env_method("get_env_stats")[0]
            reward_params = env.env_method("get_reward_config")[0]
            
                
            if current_resume_folder is not None or recovery_count > 0:

                experiment_info = old_meta.get("experiment_info", {})
                restart_history = experiment_info.get("restart_history", [])

                # --- 3. LOGIC TO PREVENT DUPLICATES AND FIX TIMELINE ---
                
                # Check if this exact timestep is already the latest entry
                if restart_history and isinstance(restart_history, list) and restart_history[-1].get("timesteps_at_restart") == timesteps_completed:
                    print("Duplicate restart entry detected. Skipping append.")
                else:
                    if not isinstance(restart_history, list): 
                        restart_history = []
                        
                    # Remove any restart entries that are ahead of our current progress 
                    restart_history = [
                        entry for entry in restart_history 
                        if entry.get("timesteps_at_restart", 0) < timesteps_completed
                    ]

                    # --- 3B. Synchronize Snapshots (Dictionaries, NOT Lists) ---
                    
                    # 1. Periodic Tracking
                    periodic_data = old_meta.get("periodic_tracking", {})
                    if not isinstance(periodic_data, dict): 
                        periodic_data = {}
                    
                    
                    periodic_tracking_latest = periodic_data.get("latest", {})
                    periodic_tracking_first_5000 = periodic_data.get("first_5000_steps", {})

                    # If 'latest' is from a future timeline that we just rolled back from, wipe it
                    if periodic_tracking_latest.get("step", 0) > timesteps_completed:
                        print(f"[Timeline Sync] Purging 'latest' tracking snapshot from future step {periodic_tracking_latest.get('step')}.")
                        periodic_tracking_latest = {}

                    # first_5000_steps is safely kept as long as we are past step 5000
                    if periodic_tracking_first_5000.get("step", 0) > timesteps_completed:
                        periodic_tracking_first_5000 = {}

                    periodic_data["latest"] = periodic_tracking_latest
                    periodic_data["first_5000_steps"] = periodic_tracking_first_5000

                    # 2. Evaluation Stats (Best Model)
                    evaluation_stats = old_meta.get("best_model_stats", {})


                
                    # --- 4. Append the new crash/restart information (This IS a list) ---
                    restart_history.append({
                        "timestamp": str(np.datetime64('now')),
                        "timesteps_at_restart": timesteps_completed,
                        "recovery_attempt_idx": recovery_count,
                        "injected_seed": current_seed,
                        "resumed_from_checkpoint": current_checkpoint_name
                    })

                    # 5. Update and Save
                    old_meta["periodic_tracking"] = periodic_data
                    old_meta["best_model_stats"] = evaluation_stats
                    old_meta["restart_history"] = restart_history
                    old_meta["experiment_info"] = experiment_info
                    
                    with open(orig_meta_path, 'w') as f:
                        json.dump(old_meta, f, indent=4)
                    print(f"Metadata history updated and timelines synchronized to step {timesteps_completed}.")

            else:
                restart_history = []
                periodic_data = {}
                evaluation_stats = {}
            
            common_params = [
                "learning_rate", "gamma", "batch_size", "verbose", "seed", 
                "device", "buffer_size", "learning_starts", "tau", "train_freq", 
                "gradient_steps", "n_steps", "n_epochs", "ent_coef", "gae_lambda", 
                "clip_range", "target_entropy", "policy_delay"
            ]
            extracted_params = {p: getattr(model, p) for p in common_params if hasattr(model, p)}

            env_info = {
                "map_file": "Full Dataset from HouseExpo_IRSim_Converter/IRSimDataset",
                "max_steps": env.get_attr("max_steps")[0],
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
                    "current_seed": current_seed,
                    "total_timesteps": args.total_timesteps,
                    "num_envs": args.num_envs,
                    "num_eval_envs": NUM_EVAL_ENVS,
                    "status": "started" if not current_resume_folder else "resumed",
                    "restarts": recovery_count,
                    "start_time": str(np.datetime64('now')),
                    "resumed_from": current_checkpoint_name if current_resume_folder else None
                },

                "restart_history": restart_history,
                "periodic_tracking": periodic_data,
                "best_model_stats": evaluation_stats,

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
            if current_resume_folder is None or recovery_count == -1:
                meta_name = "metadata.json"
                with open(os.path.join(experiment_folder, meta_name), "w") as f:
                    json.dump(metadata, f, indent=4, default=str)
            else:
                meta_name = "metadata.json"


            print(f"\n[Metadata] Configuration saved to {os.path.join(experiment_folder, meta_name)}")
            print(f"[Training] Starting {args.total_timesteps} timesteps...")

            # Set reset_num_timesteps based on whether we are resuming or not
            # Calculate remaining timesteps
            if current_resume_folder:
                # model.num_timesteps is automatically restored when loading the model in SB3
                remaining_steps = max(0, args.total_timesteps - model.num_timesteps)
                do_reset_timesteps = False
            else:
                remaining_steps = args.total_timesteps
                do_reset_timesteps = True

            if remaining_steps > 0:
                print(f"[Training] Resuming for {remaining_steps} remaining timesteps...")
                model.learn(total_timesteps=remaining_steps, callback=callback_list, reset_num_timesteps=do_reset_timesteps)
            else:
                print("[Training] Total timesteps already reached. Exiting.")
                break # Exit the recovery loop if we're already done

            # Save Final Model
            model.save(os.path.join(experiment_folder, f"model_final"))
            vec_env = model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(os.path.join(experiment_folder, f"model_final.pkl"))

            print(f"\nTraining Complete. Experiment folder: {experiment_folder}")


        except (EOFError, BrokenPipeError, Exception) as e:
            recovery_count += 1
            
            # Determine if this was an EOF/Subprocess disconnect or another error
            if isinstance(e, (EOFError, BrokenPipeError)):
                crash_type = "[EOF / WORKER DISCONNECT INTERCEPTED]"
            else:
                crash_type = "[CRASH INTERCEPTED]"

            print(f"\n{'!'*60}\n{crash_type} Training loop halted on recovery attempt {recovery_count}/{max_recovery_attempts}!")
            print(f"Exception details: {e}")
            traceback.print_exc()
            print(f"{'!'*60}\n")
            
            # Check if we hit the ceiling limit of maximum recovery allowances
            if recovery_count >= max_recovery_attempts:
                print(f"[CRITICAL] Maximum recovery attempts ({max_recovery_attempts}) reached. Hard shutdown.")
                sys.exit(1) # Note: The finally block WILL still run before sys.exit completes!
            
            # Log crash details explicitly to file
            crash_log_file = os.path.join(experiment_folder, "irsim_auto_recovery_log.txt")
            with open(crash_log_file, "a") as log:
                log.write(f"=== CRASH TIMESTAMP: {np.datetime64('now')} (Attempt #{recovery_count}) ===\n")
                log.write(f"Type: {crash_type}\n")
                traceback.print_exc(file=log)
                log.write("\n" + "="*50 + "\n\n")
            
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

        except KeyboardInterrupt:
            # Safely catch Ctrl+C to prevent orphaned background processes
            print("\n[Manual Stop] KeyboardInterrupt detected. Safely aborting training loop...")
            break # Exit the while loop and let the finally block clean up

        finally:
            # ==========================================================
            # THE GUARANTEED CLEANUP BLOCK
            # Runs on Success, on Crash, on Ctrl+C, AND right before sys.exit()
            # ==========================================================
            print("[Cleanup] Ensuring all simulation backends and workers are safely terminated...")
            try:
                if env is not None:
                    print("          -> Terminating training subprocess workers...")
                    env.close() 
            except Exception as cleanup_error:
                pass # Catch and ignore the broken pipe on close

            try:
                if eval_env is not None:
                    print("          -> Terminating validation environment context...")
                    eval_env.close()
            except Exception as cleanup_error:
                pass

if __name__ == '__main__':
    main()