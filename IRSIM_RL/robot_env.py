import gc
import json
import logging


import glob
import math
import matplotlib
matplotlib.use('Agg')
import random
import warnings 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import irsim
import os
import sys
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import binary_dilation, distance_transform_edt




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MAP_DIR = os.path.abspath(os.path.join(PROJECT_ROOT,"HouseExpo_IRSim_Converter","IRSimDataset"))

from Tracking.target_pf import Target

MAX_NODES = 100000

# ==========================================
# Gymnasium Environment definition
# ==========================================



class RobotNavEnv(gym.Env):

    def __init__(self, render=False, world_file="robot_world.yaml", pf_active=False, seed=0, 
                 is_eval=False, is_testing=False, is_serial_eval=True, worker_id=0, num_workers=1, num_eval_episodes=8):
        super(RobotNavEnv, self).__init__()

        self.seed = seed
        self.is_eval = is_eval 
        self.is_testing = is_testing 
        self.is_serial_eval = is_serial_eval 
        self.worker_id = worker_id

        # 1. SEED DISTRIBUTION
        if self.is_eval and not self.is_serial_eval:
            master_seeds = [self.seed + i for i in range(num_eval_episodes)]
            # Distributed parallel evaluation seeds
            self.eval_seeds = [master_seeds[i] for i in range(len(master_seeds)) if i % num_workers == worker_id]
        elif self.is_eval and self.is_serial_eval:
            self.eval_seeds = [self.seed + i for i in range(num_eval_episodes)]
        
        # 2. MAP DIRECTORY SELECTION
        if self.is_eval:
            self.eval_index = 0
            map_dir = os.path.join(os.path.dirname(MAP_DIR), "EvalDataset")
        elif self.is_testing:
            map_dir = os.path.join(os.path.dirname(MAP_DIR), "TestDataset")
        else:
            map_dir = os.path.join(os.path.dirname(MAP_DIR), "IRSimDataset")
            
        grid_dir = os.path.join(os.path.dirname(MAP_DIR), "OccupancyGrids")
        self.active_grid_dir = grid_dir 

        # 3. MAP DISCOVERY & WORKER SLICING
        all_maps = sorted(glob.glob(os.path.join(map_dir, "*.yaml")))
        if not all_maps:
            raise ValueError(f"Nessun file .yaml trovato in {map_dir}")
        
        if self.is_eval and not self.is_serial_eval and num_workers > 1:
            self.map_files = all_maps[worker_id::num_workers]
        else:
            self.map_files = all_maps

        # Safety check: ensure this specific worker actually has a map to load
        if not self.map_files:
            raise ValueError(f"Worker {worker_id} was assigned 0 maps from {map_dir}. "
                             f"Total maps available: {len(all_maps)}. Reduce your num_workers.")

        # 4. SIMULATION AND NAVIGATION STATE
        self.render_mode = render 
        self.current_map_name = None
        self.sim = None
        self.occupancy_grid = None
        self.map_meta = None
        self.robot_radius = 0.15

        self.map_loaded = False  
        
        # 5. TARGET TRACKING & PARTICLE FILTER
        self.pf_active = pf_active
        if pf_active:
            self.target_tracker = Target(method='range', max_pf_range=15)
            self.pf = self.target_tracker.pf
            self.particle_plot = None
            self.pf.set_validation_callback(self.is_valid_pos)  # Ensure PF never considers invalid positions
        else:
            self.target_tracker = Target(method='range')

        self.uncertainty_threshold = 0.1
        self.rew_err_th = 0.01
        self.rew_dis_th = 1.4
        self.set_max_range = 15
        self.reachedGoal = False
        self.accumulated_episode_reward = 0.0 

        # 6. REWARD WEIGHTS (Cleaned duplicates)
        self.waypoint_reward = 1.0
        self.goal_reward = 15.0
        self.w_progress = 2.0 
        self.w_rotation = 0.05          #Before 0.02
        self.w_wiggle = 0.08            #Before 0.05
        
        self.w_obstacle_front = 10.0  
        self.w_obstacle_side = 5.0    
        self.safe_dist_front = 0.20
        self.safe_dist_side = 0.10  

        self.standoff_dist = 0.5
        self.standoff_margin = 0.15
        self.heading_margin = 0.2
        self.w_heading = 2.0

        self.collision_penalty = -10
        self.collision_goal_penalty = -5
        self.truncation_penalty = -5
        self.step_penalty = -0.005 

        # 7. ALGORITHM & SPACES CONFIG
        self.state = "SEARCHING"
        self.current_path = []
        self.waypoint_index = 0
        self.nav_goal = np.array([0.0, 0.0]) 
        self.state_dim = 51
        self.max_steps = 500  
        self.astar_cooldown = 60  
        
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.2], dtype=np.float32),  
            high=np.array([0.6,  1.2], dtype=np.float32),   
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        self.estimated_goal = np.array([0.0, 0.0]) 
        self.estimated_goal_ls = np.array([0.0, 0.0]) 
        self.ghost_plot = None
        self.map_sequence_idx = 0
  
        self._reset_episode_tracking(start_position = None)

    def render(self):
            self.sim.render()
            self.sim.reset_plot()
            
            # ==========================================
            # NATIVE OCCUPANCY GRID RENDERING (FOOLPROOF)
            # ==========================================
            # if self.inflated_grid is not None:
            #     # 1. Extract and cache coordinates ONCE per map to keep FPS high
            #     if getattr(self, '_rendered_grid_map', None) != self.current_map_name:
            #         res = self.map_meta['resolution']
            #         
            #         # Find all (y, x) indices where the grid is an obstacle (True/1)
            #         # Note: swap `self.inflated_grid` with `self.occupancy_grid` to see thin walls
            #         gy, gx = np.where(self.inflated_grid) 
            #         
            #         # Convert grid indices to real-world meter coordinates
            #         # Add (res / 2.0) to center the dot directly in the middle of the cell
            #         pts_x = (gx * res) + (res / 2.0)
            #         pts_y = (gy * res) + (res / 2.0)
            #         
            #         # Stack into a list of [[x, y], [x, y]] coordinates
            #         self._grid_points = np.stack((pts_x, pts_y), axis=-1).tolist()
            #         self._rendered_grid_map = self.current_map_name
            # 
            #     # 2. Draw the cached points using irsim's proven native method
            #     if hasattr(self, '_grid_points') and len(self._grid_points) > 0:
            #         # Use a light purple ('violet') and small size (s=2) so it looks like a grid
            #         self.sim.draw_points(points=self._grid_points, c='grey', s=2)
         
            # --- A* path ---
            if self.state == "FOLLOWING" and len(self.current_path) > 0:
                path_points = self.current_path
                self.sim.draw_points(points=path_points, c='green', s=10)
                # Highlight current target waypoint
                target_wp = self.current_path[self.waypoint_index]
                self.sim.draw_points(points=[target_wp], c='yellow', s=50)

            # --- GHOST AND LS POINTS ---
            if self.pf_active:
                # (Keep your existing PF ghost rendering here)
                if hasattr(self, 'estimated_goal'):
                    gx, gy = self.estimated_goal
                    self.sim.draw_points(points=[[gx, gy]], c='magenta', s=200)
            else:
                # Render the LS Ghost (Red)
                if hasattr(self, 'estimated_goal_ls'):
                    gx, gy = self.estimated_goal_ls
                    self.sim.draw_points(points=[[gx, gy]], c='red', s=200)

                # RENDER LS OBSERVER POINTS
                # These are the robot positions used to calculate the estimate
                if hasattr(self.target_tracker, 'eastingpoints_LS'):
                    ex = self.target_tracker.eastingpoints_LS
                    ey = self.target_tracker.northingpoints_LS
                    
                    if len(ex) > 0:
                        # Ensure we are passing float values, not numpy types
                        obs_points = [[float(x), float(y)] for x, y in zip(ex, ey)]
                        self.sim.draw_points(points=obs_points, c='cyan', s=5)
                    

            # PARTICLES
            if self.pf_active:
                if hasattr(self.pf, 'x') and self.pf.x is not None:
                    px = self.pf.x[:, 0]
                    py = self.pf.x[:, 2]
                    points = np.stack([px, py], axis=1).tolist()
                    self.sim.draw_points(points=points, c='cyan', s=5)

    def close(self):
        """Ensure all resources are freed when the environment is closed."""
        # 1. Shut down the irsim backend using the correct .end() method
        if hasattr(self, 'sim') and self.sim is not None:
            try:
                self.sim.end(ending_time=0.0)
            except Exception as e:
                print(f"Warning: Failed to cleanly close irsim instance during env.close(): {e}")
    
        # 2. Excellent addition: This prevents massive RAM leaks if irsim 
        # leaves detached Matplotlib figures floating in the background!
        try:
            plt.close('all')
        except ImportError:
            pass # Just in case plt isn't imported at the top of your file
            
        # 3. Call the parent Gym/Gymnasium class closer
        super().close()

    def _reset_episode_tracking(self, start_position=None):
        """Initializes or resets all volatile variables that change per episode."""
        self.time = 0
        self.total_distance = 0.0
        self.total_velocity = 0.0
        self.accumulated_episode_reward = 0.0
        self.astar_cooldown = 0
        self.first_step = True
        self.prev_action = np.zeros(2)

        self.last_position = None
        self.last_theta = None
        
        # --- NEW: Reset success/failure flags ---
        self.success = False 
        self.failure = False
        self.crashed_into_goal = False
        self.done = False
        
        if start_position is not None:
            self.last_position = start_position.copy()
        else:
            self.last_position = None

    def _get_episode_info(self, terminal, reward):
        # Calculate current target error (Ground Truth vs Estimate)
        current_estimate = self.estimated_goal if self.pf_active else self.estimated_goal_ls
        real_goal = self.sim.robot.goal if hasattr(self.sim, 'robot') else self.sim.get_robot_info(0).goal
        target_error = np.linalg.norm(current_estimate - real_goal[:2, 0])

        avg_velocity = self.total_velocity / self.time if self.time > 0 else 0.0
        
        return {
            'success': 1.0 if self.success else 0.0,
            'collision': 1.0 if self.failure else 0.0,
            'goal_crash': 1.0 if self.crashed_into_goal else 0.0,
            'reward': reward, 
            'target_error': target_error, 
            'total_distance': self.total_distance,
            'average_velocity': avg_velocity, # Now properly included
            'steps': self.time 
        }
    
    def get_reward_config(self):
        """Returns the reward weights and thresholds used in the current session."""
        return {
            "reward_type": "hybrid_switching_state_machine",
            "sparse_rewards": {
                "waypoints": self.waypoint_reward,
                "goal": self.goal_reward,
            },
            "sparse_penalties": {
                "collision": self.collision_penalty,
                "truncation": self.truncation_penalty
            },
            "dense_rewards": {
                "progress_formula": "progress = (self.prev_dist - dist_ghost) * self.w_progress",
                "progress_multiplier": self.w_progress,
            },
            "dense_penalties": {
                "time_step": self.step_penalty,

                "safe_distance_front_formula": "front_penalty = -self.w_obstacle_front * (self.safe_dist_front - min_front_dist)",
                "safe_distance_front": self.safe_dist_front,
                "w_obstacle_front": self.w_obstacle_front,

                "safe_distance_side_formula": "side_penalty = -self.w_obstacle_side * (self.safe_dist_side - min_side_dist)",
                "safe_distance_side": self.safe_dist_side,
                "w_obstacle_side": self.w_obstacle_side,

                "rotation_penalty_formula": "rotation_penalty = -self.w_rotation * abs(angular_velocity)",
                "w_rotation": self.w_rotation,

                "wiggle_penalty_formula": "wiggle_penalty = -self.w_wiggle * action_diff",
                "w_wiggle": self.w_wiggle
            },
        }

    def get_env_stats(self):
        return {
            "uncertainty_threshold": self.uncertainty_threshold,
            "max_steps": self.max_steps,
            "state_dimension": self.state_dim,
            "action_space_low": self.action_space.low.tolist(),
            "action_space_high": self.action_space.high.tolist(),
            "A* max nodes": MAX_NODES, 
            "dt": getattr(self, "dt", 0.1),
            
        }
    
    def _calculate_metrics(self, current_position, action):
        if self.last_position is not None:
            step_distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += step_distance
            self.total_velocity += np.linalg.norm(action)
        self.last_position = current_position   
    
    def get_astar_path(self, start_m, goal_m, max_nodes=MAX_NODES):
        res = self.map_meta['resolution']
        
        start_grid = (int(start_m[0]/res), int(start_m[1]/res))
        goal_grid = (int(goal_m[0]/res), int(goal_m[1]/res))

        # 1. FIXED: Boundary check to prevent IndexError
        def is_clear(gx, gy):
            if not (0 <= gy < self.inflated_grid.shape[0] and 0 <= gx < self.inflated_grid.shape[1]):
                return False
            return self.inflated_grid[gy, gx] == 0

        # If the start or goal is inside an obstacle, fail early to save CPU
        if not is_clear(start_grid[0], start_grid[1]) or not is_clear(goal_grid[0], goal_grid[1]):
            return None

        # queue: (f_score, g_score, current_node)
        queue = [(0, 0, start_grid)] 
        visited = {start_grid: 0}
        parent_map = {} 

        nodes_visited = 0
        while queue:
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return None 
            
            _, g_score, current = heapq.heappop(queue)
            
            if current == goal_grid:
                # Reconstruct path by walking backwards from goal
                path = []
                curr = current
                while curr in parent_map:
                    path.append([curr[0]*res, curr[1]*res])
                    curr = parent_map[curr]
                path.append([start_grid[0]*res, start_grid[1]*res])
                return path[::-1] # Reverse it
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if is_clear(neighbor[0], neighbor[1]):
                    step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    new_g_score = g_score + step_cost
                    
                    if neighbor not in visited or new_g_score < visited[neighbor]:
                        visited[neighbor] = new_g_score
                        parent_map[neighbor] = current 
                        
                        # 2. OPTIMIZED: Slight heuristic inflation (1.001) for faster tie-breaking
                        h = math.hypot(neighbor[0]-goal_grid[0], neighbor[1]-goal_grid[1]) * 1.001
                        heapq.heappush(queue, (new_g_score + h, new_g_score, neighbor))
                            
        return None
    

    def prune_path_to_sparse(self, dense_path, min_distance=0.5):
        """
        Saves CPU and guarantees safety by connecting waypoints 
        as long as there is a clear Line-of-Sight (LOS) between them.
        Then clusters waypoints that are too close together.
        """
        if not dense_path:
            return []
        if len(dense_path) < 3:
            return dense_path[1:]

        # STEP 1: LOS Pruning
        sparse_path = [dense_path[0]]
        curr_idx = 0

        while curr_idx < len(dense_path) - 1:
            next_idx = curr_idx + 1
            
            for look_ahead in range(curr_idx + 2, len(dense_path)):
                if self._check_line_of_sight(dense_path[curr_idx], dense_path[look_ahead]):
                    next_idx = look_ahead
                else:
                    break
            
            sparse_path.append(dense_path[next_idx])
            curr_idx = next_idx

        # STEP 2: Remove redundant waypoints in small areas
        # Drop the starting node first
        waypoints_only = sparse_path[1:] if len(sparse_path) > 1 else []
        
        if len(waypoints_only) < 2:
            return waypoints_only
        
        # Cluster waypoints that are too close
        clusters = []
        current_cluster = [waypoints_only[0]]
        
        for i in range(1, len(waypoints_only)):
            current_wp = waypoints_only[i]
            last_wp = current_cluster[-1]
            
            distance = np.linalg.norm(np.array(current_wp) - np.array(last_wp))
            
            if distance < min_distance:
                # Add to current cluster
                current_cluster.append(current_wp)
            else:
                # Finalize current cluster by taking its centroid
                if current_cluster:
                    centroid = np.mean(current_cluster, axis=0)
                    clusters.append(centroid.tolist())
                # Start new cluster
                current_cluster = [current_wp]
        
        # Don't forget the last cluster
        if current_cluster:
            centroid = np.mean(current_cluster, axis=0)
            clusters.append(centroid.tolist())
        
        return clusters

    def _check_line_of_sight(self, pt1, pt2):
        """Samples points along a line to ensure no walls are crossed."""
        p1, p2 = np.array(pt1), np.array(pt2)
        dist = np.linalg.norm(p2 - p1)
        
        if dist == 0:
            return True
            
        res = self.map_meta['resolution']
        step_size = res / 2.0  # Sample twice per grid cell to be absolutely safe
        steps = int(dist / step_size)
        
        # If points are very close, just check the endpoints
        if steps <= 1:
            return self.is_valid_pos(p1[0], p1[1]) and self.is_valid_pos(p2[0], p2[1])
            
        for i in range(1, steps):
            t = i / steps
            # Linear interpolation between the two points
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            
            if not self.is_valid_pos(x, y):
                return False
                
        return True

    def _load_map(self, yaml_path, grid_dir):
        """Helper function to load a new simulator instance and its corresponding grid mapping."""
        base_name = os.path.basename(yaml_path).replace(".yaml", "")
        self.grid_dir = grid_dir
        
        # 1. FIXED: Added safety check to ensure grids actually exist before skipping
        if self.current_map_name == base_name and hasattr(self, 'inflated_grid'):
            return
            
        self.current_map_name = base_name
        
        # --- FIXED: Destroy the old simulator instance cleanly ---
        if hasattr(self, 'sim') and self.sim is not None:
            try:
               # 1. Use the native Gym alias, but OVERRIDE the 3-second delay bomb
                self.sim.end(ending_time=0.0) 
                
                # 2. Delete the Python object and force garbage collection
                del self.sim
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to cleanly close old irsim instance: {e}")
                        
        self.sim = irsim.make(yaml_path, display=self.render_mode)

        if hasattr(self.sim, 'set_random_seed'):
            self.sim.set_random_seed(self.seed)

        # Load grid and metadata    
        grid_path = os.path.join(self.grid_dir,f"{base_name}.npy")
        meta_path = os.path.join(self.grid_dir, f"{base_name}_meta.json")
        
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid missing for {base_name}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata json missing for {base_name}")
            
        raw_grid = np.load(grid_path)
        with open(meta_path, "r") as f:
            self.map_meta = json.load(f)

        # Calculate robot radius in pixels
        res = self.map_meta['resolution']
        safety_margin = 0.10  # 10 cm buffer
        pixel_radius = int(np.ceil((self.robot_radius + safety_margin) / res))
        
        # Create a circular kernel for dilation
        y, x = np.ogrid[-pixel_radius:pixel_radius+1, -pixel_radius:pixel_radius+1]
        mask = x**2 + y**2 <= pixel_radius**2
        
        # 3. OPTIMIZED: Cast to int8 to play perfectly with your A* `== 0` checks
        dilated_bool = binary_dilation(raw_grid, structure=mask)
        self.inflated_grid = dilated_bool.astype(np.int8)
        self.occupancy_grid = raw_grid.astype(np.int8)
    

    def is_valid_pos(self, x, y):
        res = self.map_meta['resolution']
        gx, gy = int(x / res), int(y / res)
        
        # Fast O(1) bounds and occupancy check
        if 0 <= gx < self.inflated_grid.shape[1] and 0 <= gy < self.inflated_grid.shape[0]:
            return not self.inflated_grid[gy, gx]
        return False


    def _get_random_valid_pos(self, margin=0.3):
        """Uses the occupancy grid to find a truly navigable spawn point."""
        map_obj = self.sim.get_map()
        w, h = map_obj.width, map_obj.height
        
        # Increased to 500 to account for denser obstacle maps
        max_attempts = 500 
        for _ in range(max_attempts):
            x = float(self.np_random.uniform(0.5, w - 0.5))
            y = float(self.np_random.uniform(0.5, h - 0.5))
            
            # Check 1: Is it in the 'White' area of the grid?
            if self.is_valid_pos(x, y):
                
                # Check 2: Is it physically clear (Lidar/Collision)?
                self.sim.robot.set_state(np.array([[x], [y], [0.0]]))
                self.sim.step(np.array([[0.0], [0.0]]))
                
                scan = self.sim.get_lidar_scan()
                ranges = scan["ranges"] if isinstance(scan, dict) else scan
                
                # MISSING LOGIC ADDED: Ensure the closest obstacle is further than the margin
                if np.min(ranges) >= margin:
                    return x, y
        
        logging.error("Failed to find valid pos after %d attempts.", max_attempts)
        # It is much safer to raise an exception and let your environment 
        # reset/handle it rather than blindly spawning inside a wall.
        return None, None
                
            
        
    def reward(self, data):
        latest_scan, dist_ghost, cos, sin, collision, arrive, diff_rad, action, real_goal, true_v, true_w = data
        
        # --- 1. Base Time Penalty ---
        total_rew = self.step_penalty 

        diff_rad_norm = (diff_rad + np.pi) % (2 * np.pi) - np.pi
        
        # --- 2. State-Aware Dense Reward (Progress & Heading) ---
        if self.state == "DOCKING":
            # 2a. Standoff Distance Progress
            # We want dist_ghost to reach self.standoff_dist, NOT 0.0
            current_dist_error = abs(dist_ghost - self.standoff_dist)
            prev_dist_error = abs(self.prev_dist - self.standoff_dist)
            dist_progress = prev_dist_error - current_dist_error
            
            # 2b. Heading Progress (Look at the goal)
            current_heading_error = abs(diff_rad_norm)
            if hasattr(self, 'prev_heading_error'):
                heading_progress = self.prev_heading_error - current_heading_error
                heading_progress = np.clip(heading_progress, -0.5, 0.5)
            else:
                heading_progress = 0.0
            self.prev_heading_error = current_heading_error

            # Add both to total reward
            total_rew += (dist_progress * self.w_progress) + (heading_progress * self.w_heading)
        else:
            # Normal following progress
            progress = self.prev_dist - dist_ghost
            total_rew += progress * self.w_progress 
            
            # Reset heading error so it doesn't spike when switching to DOCKING
            self.prev_heading_error = abs(diff_rad_norm)
            
        self.prev_dist = dist_ghost

        # --- 3. Stable Obstacle Avoidance ---
        num_beams = len(latest_scan)
        cone_width_in_beams = 10 
        center_idx = num_beams // 2 
        half_width = cone_width_in_beams // 2 

        front_start = center_idx - half_width 
        front_end = center_idx + half_width   

        front_cone = latest_scan[front_start : front_end]
        side_cones = np.concatenate([latest_scan[:front_start], latest_scan[front_end:]])

        min_front_dist = float(min(front_cone))
        min_side_dist = float(min(side_cones))

        if min_front_dist < self.safe_dist_front:
            front_penalty = -self.w_obstacle_front * (self.safe_dist_front - min_front_dist)
            total_rew += front_penalty

        if min_side_dist < self.safe_dist_side:
            side_penalty = -self.w_obstacle_side * (self.safe_dist_side - min_side_dist)
            total_rew += side_penalty

        # --- 4. Wiggle & Smoothness Penalty ---
        angular_velocity = action[1] 
        rotation_penalty = -self.w_rotation * abs(angular_velocity)

        if hasattr(self, 'prev_action'):
            action_diff = np.sum(np.abs(np.array(action) - np.array(self.prev_action)))
            wiggle_penalty = -self.w_wiggle * action_diff
        else:
            wiggle_penalty = 0.0

        total_rew += (rotation_penalty + wiggle_penalty)
        self.prev_action = np.copy(action)

        # --- 5. Milestone & Terminal Logic ---
        is_final_stage = self.state == "DOCKING"

        if collision:
            total_rew = self.collision_penalty
            self.failure = True
            self.done = True

        elif is_final_stage:
            # Check Standoff Criteria
            at_standoff = abs(dist_ghost - self.standoff_dist) <= self.standoff_margin
            facing_goal = abs(diff_rad_norm) <= self.heading_margin

            # Remove the action checks and terminate strictly on position
            if at_standoff and facing_goal:
                total_rew += self.goal_reward  
                self.success = True
                self.done = True
                
            elif dist_ghost < (self.standoff_dist - self.standoff_margin):
                # Penalty for breaching the standoff zone (getting too close)
                # This prevents it from accidentally ramming the target
                total_rew -= self.collision_goal_penalty
                self.crashed_into_goal = True
                self.done = True

        elif not is_final_stage and dist_ghost < 0.5:
            total_rew += self.waypoint_reward  

        self.accumulated_episode_reward += total_rew
        return total_rew
    
    def _extract_sim_data(self, action):
        scan = self.sim.get_lidar_scan()
        latest_scan = scan["ranges"] if isinstance(scan, dict) else scan
        robot_state = self.sim.get_robot_state()  # [x, y, theta]
        
        # --- Compute Ground Truth Kinematics ---
        dt = getattr(self, "dt", 0.1)
        current_pos = np.array([robot_state[0, 0], robot_state[1, 0]])
        current_theta = robot_state[2, 0]
        
        if self.last_position is not None and self.time > 0:
            # Linear velocity: distance moved over time
            true_v = np.linalg.norm(current_pos - self.last_position) / dt
            # Angular velocity: delta theta normalized to [-pi, pi] over time
            diff_theta = (current_theta - self.last_theta + np.pi) % (2 * np.pi) - np.pi
            true_w = diff_theta / dt
        else:
            true_v = 0.0
            true_w = 0.0
            
        # Cache for next iteration's velocity calculation
        self.last_position = current_pos
        self.last_theta = current_theta
        
        
        # Keep your existing target/ghost calculations...
        if hasattr(self.sim, 'robot'):
            real_goal = np.array([self.sim.robot.goal[0,0], self.sim.robot.goal[1,0]])
            collision = self.sim.robot.collision
            arrive = self.sim.robot.arrive
        else:
            info = self.sim.get_robot_info(0)
            real_goal = np.array([info.goal[0,0], info.goal[1,0]])
            collision = info.collision
            arrive = info.arrive

        goal_vector = [self.nav_goal[0] - robot_state[0,0], self.nav_goal[1] - robot_state[1,0]]
        distance_to_ghost = np.linalg.norm(goal_vector)
        
        norm_pose = np.array([np.cos(robot_state[2,0]), np.sin(robot_state[2,0])])
        norm_ghost = goal_vector / (distance_to_ghost + 1e-6)
        
        cos_val = np.dot(norm_pose, norm_ghost)
        sin_val = np.cross(norm_pose, norm_ghost)
        diff_rad = np.arctan2(sin_val, cos_val)
        
        # Append true_v and true_w to the returned data tuple
        return latest_scan, distance_to_ghost, cos_val, sin_val, collision, arrive, diff_rad, action, real_goal, true_v, true_w
                
    def prepare_state(self, data):
        # Unpack the 11 items now being returned
        latest_scan, distance, cos, sin, collision, arrive, diff_rad, action, real_goal, true_v, true_w = data
        
        scan_arr = np.array(latest_scan)
        scan_arr = np.clip(scan_arr, 0, self.set_max_range)
        
        state_encoded = [
            1.0 if self.state == "SEARCHING" else 0.0,
            1.0 if self.state == "FOLLOWING" else 0.0,
            1.0 if self.state == "DOCKING" else 0.0
        ]
        
        clipped_dist = np.clip(distance, 0, self.set_max_range)
        
        # Add true physical metrics alongside your control inputs
        extra_features = [
            clipped_dist / self.set_max_range, 
            cos, 
            sin, 
            action[0] / 0.6,    # Intended linear
            action[1] / 1.2,    # Intended angular
            true_v / 0.6,       # REAL physical linear velocity (Normalized [0, 1])
            true_w / 1.2,       # REAL physical angular velocity (Normalized [-1, 1])
            *state_encoded
        ]

        num_extra = len(extra_features)
        max_bins = self.observation_space.shape[0] - num_extra
        
        bins = np.array_split(scan_arr, max_bins)
        min_values = 1.0 - (np.array([np.min(b) for b in bins]) / self.set_max_range)

        full_state = np.concatenate([min_values, np.array(extra_features, dtype=np.float32)])
        terminal = collision or arrive
        
        return full_state, terminal
    
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- MAP SKIPPING SAFEGUARD ---
        max_skips = 10
        for attempt in range(max_skips):
            
            # 1. Determine Seed and Map Index
            if self.is_eval:
                active_seed = self.eval_seeds[self.eval_index]
                random.seed(active_seed)
                np.random.seed(active_seed)
                
                idx = self.eval_index % len(self.map_files)
                # Advance the index for the next attempt (or next episode)
                self.eval_index = (self.eval_index + 1) % len(self.eval_seeds)
                
            else:
                if seed is not None:
                    active_seed = seed
                else:
                    active_seed = int(self.np_random.integers(0, 2**31 - 1))
                random.seed(active_seed)
                np.random.seed(active_seed)
                
                idx = self.np_random.integers(0, len(self.map_files))

            if hasattr(self.sim, 'set_random_seed'):
                self.sim.set_random_seed(active_seed)

            # 2. Load the Map
            selected_yaml = self.map_files[idx]
            self._load_map(
                selected_yaml, 
                grid_dir=self.active_grid_dir  
            )
            self.sim.reset()

            # 3. Randomize Robot Position
            rx, ry = self._get_random_valid_pos(margin=0.5)
            
            # CATCH 1: Robot spawn failed
            if rx is None:
                logging.warning(f"Attempt {attempt+1}: Map {selected_yaml} rejected (Robot spawn). Skipping...")
                continue # Loop back and try the next map/seed
                
            new_theta = self.np_random.uniform(-np.pi, np.pi)

            # 4. Randomize Goal Position - PREFER DIFFERENT ROOM (Broken LOS)
            max_attempts = 100
            best_fallback_gx, best_fallback_gy = None, None
            goal_spawn_failed = False
            
            for _ in range(max_attempts):
                gx, gy = self._get_random_valid_pos(margin=0.5)
                
                # CATCH 2: Goal spawn failed completely
                if gx is None:
                    goal_spawn_failed = True
                    break # Break the goal loop
                
                dist = np.sqrt((rx - gx)**2 + (ry - gy)**2)
                
                if dist > 1.0:
                    if best_fallback_gx is None:
                        best_fallback_gx, best_fallback_gy = gx, gy
                    
                    if not self._check_line_of_sight([rx, ry], [gx, gy]):
                        break 
            else:
                if best_fallback_gx is not None:
                    gx, gy = best_fallback_gx, best_fallback_gy
                else:
                    gx, gy = rx + 0.5, ry + 0.5

            if goal_spawn_failed:
                logging.warning(f"Attempt {attempt+1}: Map {selected_yaml} rejected (Goal spawn). Skipping...")
                continue # Loop back and try the next map/seed

            # If we reach this line, both robot and goal spawned successfully!
            break # Break out of the map skipping loop
            
        else:
            # This executes ONLY if the for-loop exhausts all 10 attempts without breaking
            raise RuntimeError(f"Failed to spawn after {max_skips} consecutive maps. Check your dataset, scale, or margin parameters.")

        # --- ENVIRONMENT SETUP (Proceeds normally with valid coordinates) ---
        self.sim.robot.set_state(np.array([[rx], [ry], [new_theta]]))
        self.sim.robot.set_goal(np.array([[gx], [gy], [0.0]]))

        # Get the real robot and goal states
        robot_state = self.sim.get_robot_state()


        real_goal = self.sim.robot.goal if hasattr(self.sim, 'robot') else self.sim.get_robot_info(0).goal
        
        # Format robot position for the PF: [x, vx, y, vy]
        robot_pos_pf = [robot_state[0,0], 0.0, robot_state[1,0], 0.0]
        initial_dist = np.linalg.norm(robot_state[:2, 0] - real_goal[:2, 0])
        self.best_dist = initial_dist
        
        if self.pf_active:
            self.pf.init_particles(position=robot_pos_pf, slantrange=initial_dist)
            self.target_tracker.updatePF(dt=0.1, new_range=True, z=initial_dist, myobserver=robot_pos_pf)
            self.estimated_goal = np.array([self.target_tracker.pfxs[0], self.target_tracker.pfxs[2]])
        else:
            self.target_tracker.allz = []
            self.target_tracker.eastingpoints_LS = []
            self.target_tracker.northingpoints_LS = []
            self.target_tracker.lsxs = []
            self.target_tracker.updateLS(dt=0.1, new_range=True, z=initial_dist, myobserver=robot_pos_pf)
            if len(self.target_tracker.lsxs) > 0:
                self.estimated_goal_ls = np.array([self.target_tracker.lsxs[-1][0], self.target_tracker.lsxs[-1][2]])
            
        self.p_pos_origin = robot_state[:2, 0].copy()

        # State machine reset configuration
        self.state = "SEARCHING"
        self.current_path = []
        self.waypoint_index = 0
        self.nav_goal = self.estimated_goal.copy() if self.pf_active else self.estimated_goal_ls.copy()

        # Force IRSIM to update sensors at the new location
        self.sim.step(np.array([[0.0], [0.0]])) 
        
        # 1. Unpack the tuple immediately to make Pylance and Python happy
        sim_data = self._extract_sim_data(action=[0.0, 0.0])
        (
            latest_scan, distance, cos, sin, 
            collision, arrive, diff_rad, action, real_goal, true_v, true_w
        ) = sim_data

        # 2. Set your tracking history variables using the unpacked data
        self.prev_heading_error = diff_rad
        self.prev_dist = distance  # No need to recalculate this manually below anymore!

        obs, terminal = self.prepare_state(sim_data) 

        self.prev_dist = np.linalg.norm(robot_state[:2, 0] - self.nav_goal)
        
        # --- THE FIX: Pass physical tracking starting points safely ---
        self._reset_episode_tracking(start_position=robot_state[:2, 0])
        
        return obs, {}

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Environment simulation is not initialized. Did reset() fail?")
        
        # 1. Global timer decrements (Do this right away)
        if hasattr(self, 'astar_cooldown') and self.astar_cooldown > 0:
            self.astar_cooldown -= 1
        
        # Move the robot
        ctrl_action = np.array([[action[0]], [action[1]]])
        self.sim.step(ctrl_action)
        
        # --- ESTIMATION ---
        robot_state = self.sim.get_robot_state()
        robot_pos_pf = np.array([robot_state[0,0], 0.0, robot_state[1,0], 0.0])
        real_goal = self.sim.robot.goal if hasattr(self.sim, 'robot') else self.sim.get_robot_info(0).goal
        dist_z = np.linalg.norm(robot_state[:2, 0] - real_goal[:2, 0])
        
        if self.pf_active:
            self.target_tracker.updatePF(dt=0.1, new_range=True, z=dist_z, myobserver=robot_pos_pf)
            self.estimated_goal = np.array([self.target_tracker.pfxs[0], self.target_tracker.pfxs[2]])
            
            # ==========================================
            # GATING LOGIC & STATE MACHINE (PF MODE)
            # ==========================================
            if self.state == "SEARCHING":
                x_var = np.var(self.pf.x[:, 0])
                y_var = np.var(self.pf.x[:, 2])
                uncertainty = x_var + y_var
                
                self.nav_goal = self.estimated_goal.copy()

                # Only attempt pathfinding if uncertainty is low AND cooldown has expired
                if uncertainty < self.uncertainty_threshold and self.astar_cooldown == 0:
                    
                    if self.is_valid_pos(self.estimated_goal[0], self.estimated_goal[1]):
                        path = self.get_astar_path(robot_state[:2, 0], self.estimated_goal)
                        
                        if path is not None:
                            if len(path) > 1:
                                # --- SUCCESS: Path found! No cooldown needed ---
                                sparse_path = self.prune_path_to_sparse(path)
                                
                                pruned_path = []
                                for wp in sparse_path:
                                    dist_to_goal = np.linalg.norm(np.array(wp) - self.estimated_goal)
                                    if dist_to_goal > 0.3: 
                                        pruned_path.append(wp)
                                    else:
                                        break 
                                        
                                if len(pruned_path) > 0:
                                    self.current_path = pruned_path
                                    self.waypoint_index = 0
                                    self.nav_goal = np.array(self.current_path[0])
                                    self.state = "FOLLOWING"
                                else:
                                    self.state = "DOCKING"

                            else:
                                # Path is too short to be useful, go straight to docking
                                self.state = "DOCKING"
                                self.nav_goal = self.estimated_goal.copy()
                        else:
                            # --- FAILURE 1: A* calculation failed (unreachable topology) ---
                            self.astar_cooldown = 60  # Set cooldown to rest the CPU
                            self.state = "SEARCHING"
                    else:
                        # --- FAILURE 2: Goal is inside a wall geometry ---
                        self.astar_cooldown = 60  # Don't even waste CPU running A* on a wall
                        self.state = "SEARCHING"

            elif self.state == "FOLLOWING":
                target_wp = self.current_path[self.waypoint_index]
                dist_to_wp = np.linalg.norm(robot_state[:2, 0] - target_wp)

                if dist_to_wp < 0.5:
                    if self.waypoint_index < len(self.current_path) - 1:
                        self.waypoint_index += 1
                        self.nav_goal = np.array(self.current_path[self.waypoint_index])
                        self.prev_dist = np.linalg.norm(robot_state[:2, 0] - self.nav_goal)
                    else:
                        self.state = "DOCKING"
                        self.nav_goal = self.estimated_goal.copy()
                        self.current_path = []
                            
            elif self.state == "DOCKING":
                self.nav_goal = self.estimated_goal.copy()

        else: # LS MODE
            self.target_tracker.updateLS(dt=0.1, new_range=True, z=dist_z, myobserver=robot_pos_pf)
            
            if len(self.target_tracker.lsxs) > 0:
                new_ls_estimate = np.array([self.target_tracker.lsxs[-1][0], self.target_tracker.lsxs[-1][2]])
                
                if self.state == "SEARCHING":
                    num_points = len(self.target_tracker.eastingpoints_LS)
                    self.estimated_goal_ls = new_ls_estimate
                    self.nav_goal = self.estimated_goal_ls.copy()

                    # Added cooldown check to LS mode to prevent CPU throttling
                    if num_points >= 5 and self.astar_cooldown == 0: 
                        
                        if self.is_valid_pos(new_ls_estimate[0], new_ls_estimate[1]):
                            path = self.get_astar_path(robot_state[:2, 0], new_ls_estimate)
                            
                            if path is not None and len(path) > 1:
                                # --- SUCCESS: Path found! ---
                                sparse_path = self.prune_path_to_sparse(path)
                                self.current_path = sparse_path
                                self.waypoint_index = 0
                                self.nav_goal = np.array(self.current_path[0])
                                self.state = "FOLLOWING"
                            else:
                                # --- FAILURE 1: A* failed ---
                                self.astar_cooldown = 60
                        else:
                            # --- FAILURE 2: LS estimate is inside a wall ---
                            self.astar_cooldown = 60
                
                elif self.state == "FOLLOWING":
                    ls_drift = np.linalg.norm(new_ls_estimate - self.estimated_goal_ls)
                    dist_to_path_end = np.linalg.norm(new_ls_estimate - self.current_path[-1])

                    if ls_drift > 1.0 or dist_to_path_end > 1.5:
                        print("LS Estimate Jumped! Reverting to SEARCHING.")
                        self.state = "SEARCHING"
                        self.current_path = []
                        self.estimated_goal_ls = new_ls_estimate
                        self.nav_goal = self.estimated_goal_ls.copy()
                    else:
                        target_wp = self.current_path[self.waypoint_index]
                        dist_to_wp = np.linalg.norm(robot_state[:2, 0] - target_wp)

                        if dist_to_wp < 0.5:
                            # FIXED: Check if there are more waypoints left
                            if self.waypoint_index < len(self.current_path) - 1:
                                self.waypoint_index += 1
                                target_wp = self.current_path[self.waypoint_index]
                                self.nav_goal = np.array(target_wp)
                                self.prev_dist = np.linalg.norm(robot_state[:2, 0] - self.nav_goal)
                            else:
                                # FIXED: Safely transition out of FOLLOWING mode at path end
                                self.state = "DOCKING"
                                self.nav_goal = new_ls_estimate.copy()
                                self.current_path = []

                        self.estimated_goal_ls = new_ls_estimate
                
                elif self.state == "DOCKING":
                    self.nav_goal = new_ls_estimate.copy()

        # Handle tracking on the very first frame
        if getattr(self, 'first_step', True):
            self.prev_dist = np.linalg.norm(robot_state[:2, 0] - self.nav_goal)
            self.first_step = False

        # --- DATA EXTRACTION & ENVIRONMENT PIPELINE ---
        sim_data = self._extract_sim_data(action=action)
        
        # Clean execution: Separate the reward calculation from state preparation
        reward_value = self.reward(sim_data) 
        obs, terminal = self.prepare_state(sim_data) # Passes raw sim_data cleanly!

        self._calculate_metrics(robot_state[:2, 0], action)
        self.time += 1
        
        terminated = self.done
        truncated = self.time >= self.max_steps

        # Handle timeout penalty safely without overwriting collision penalties
        if truncated and not terminated:
            reward_value = self.truncation_penalty
            self.done = True  # Ensure the episode ends on timeout

        info = self._get_episode_info(terminated, reward_value)
    
        if self.render_mode:
            self.render()

        return obs, float(reward_value), terminated, truncated, info
    

