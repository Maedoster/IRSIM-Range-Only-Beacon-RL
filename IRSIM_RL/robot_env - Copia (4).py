import json
import logging

import glob
import random 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import irsim
import os
import sys
import matplotlib.pyplot as plt
import heapq


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MAP_DIR = os.path.abspath(os.path.join(PROJECT_ROOT,"HouseExpo_IRSim_Converter","IRSimDataset"))

from Tracking.target_pf import Target

MAX_NODES = 50000

# ==========================================
# Gymnasium Environment definition
# ==========================================



class RobotNavEnv(gym.Env):
    def __init__(self, render=False, world_file="robot_world.yaml", pf_active=False, seed=0):
        super(RobotNavEnv, self).__init__()

        self.seed = seed
        grid_dir = os.path.join(os.path.dirname(MAP_DIR), "OccupancyGrids")
        map_dir = os.path.join(os.path.dirname(MAP_DIR), "IRSimDataset")

        self.map_files = sorted(glob.glob(os.path.join(map_dir, "*.yaml")))
        if not self.map_files:
            raise ValueError(f"Nessun file .yaml trovato in {map_dir}")
        
        self.render_mode = render 

        self.current_map_name = None
        self.sim = None
        self.occupancy_grid = None
        self.map_meta = None

        self._load_map(self.map_files[0], grid_dir=grid_dir)
        
        if pf_active:

            self.target_tracker = Target(method='range', max_pf_range=15)
            self.pf = self.target_tracker.pf
            self.particle_plot = None
        else:
            self.target_tracker = Target(method='range')

        self.pf_active = pf_active

        self.uncertainty_threshold = 0.05

        self.rew_err_th = 0.01
        self.rew_dis_th = 1.4
        self.set_max_range = 15

        self.w_progress = 10
        self.w_rotation = 0.4
        self.safe_dist = 0.5
        self.collision_penalty = -100
        self.step_penalty = -1

        self.state = "SEARCHING"
        self.current_path = []
        self.waypoint_index = 0
        self.nav_goal = np.array([0.0, 0.0]) 
        
        self.render_mode = render
        self.state_dim = 49  
        self.max_steps = 500  
        
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.2], dtype=np.float32),  
            high=np.array([0.6,  1.2], dtype=np.float32),   
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        self.estimated_goal = np.array([0.0, 0.0]) # Initialize with zeros
        self.estimated_goal_ls = np.array([0.0, 0.0]) # Initialize with zeros
        self.ghost_plot = None
        

        if pf_active:
            self.pf.set_validation_callback(self.is_valid_particle)
  
        self._reset_episode_tracking()
    
    def get_astar_path(self, start_m, goal_m, max_nodes=MAX_NODES, robot_radius=0.3):
        res = self.map_meta['resolution']
       
        start_grid = (int(start_m[0]/res), int(start_m[1]/res))
        goal_grid = (int(goal_m[0]/res), int(goal_m[1]/res))

        # Calculate safety margin in grid cells to prevent wall-hugging
        grid_margin = int(robot_radius / res)
        grid_h, grid_w = self.occupancy_grid.shape

        # Helper function to check clearance around a grid node
        def is_clear(gx, gy):
            if not (0 <= gx < grid_w and 0 <= gy < grid_h):
                return False
            
            y_min = max(0, gy - grid_margin)
            y_max = min(grid_h, gy + grid_margin + 1)
            x_min = max(0, gx - grid_margin)
            x_max = min(grid_w, gx + grid_margin + 1)
            
            # True if no obstacles (1s) are found in the bounding patch
            return not np.any(self.occupancy_grid[y_min:y_max, x_min:x_max] == 1)

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
                
                # Replaced the single-cell check with the safe bounding-box check
                if is_clear(neighbor[0], neighbor[1]):
                    
                    step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    new_g_score = g_score + step_cost
                    
                    if neighbor not in visited or new_g_score < visited[neighbor]:
                        visited[neighbor] = new_g_score
                        parent_map[neighbor] = current # Store parent
                        
                        # Manhattan or Euclidean distance for heuristic
                        h = np.sqrt((neighbor[0]-goal_grid[0])**2 + (neighbor[1]-goal_grid[1])**2)
                        heapq.heappush(queue, (new_g_score + h, new_g_score, neighbor))
                            
        return None

    def _load_map(self, yaml_path, grid_dir):
        """Helper function per caricare un nuovo simulatore e la relativa griglia."""
        base_name = os.path.basename(yaml_path).replace(".yaml", "")
        self.grid_dir = grid_dir
        
        # If we're already on the correct map, no need to reload everything
        if self.current_map_name == base_name:
            return
            
        self.current_map_name = base_name

        if self.sim is not None:
            if hasattr(self.sim, 'close'):
                self.sim.close()
            
            plt.close('all')
        
        
        # Close old simulator if it exists to prevent memory leaks
        if self.sim is not None and hasattr(self.sim, 'close'):
            self.sim.close()
        
  
        self.sim = irsim.make(yaml_path, display=self.render_mode)

        if hasattr(self.sim, 'set_random_seed'):
            self.sim.set_random_seed(self.seed)

        # Load grid and metadata    
        grid_path = os.path.join(self.grid_dir, f"{base_name}.npy")
        meta_path = os.path.join(self.grid_dir, f"{base_name}_meta.json")
        
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid missing for {base_name}")
            
        self.occupancy_grid = np.load(grid_path)
        with open(meta_path, "r") as f:
            self.map_meta = json.load(f)

    def get_reward_config(self):
        """Returns the reward weights and thresholds used in the current session."""
        return {
            "reward_type": "hybrid_switching_state_machine",
            "thresholds": {
                "error_dist": self.rew_err_th,
                "goal_arrival": self.rew_dis_th,
                "max_lidar_range": self.set_max_range,
                "safe_obstacle_dist": self.safe_dist
            },
            "terminal_penalties": {
                "collision": self.collision_penalty,
            },
            "state_dependent_weights": {
                "SEARCHING": {"re_weight": 15, "rd_weight": 5},
                "FOLLOWING": {"re_weight": 5, "rd_weight": 15}
            },
            "action_penalties": {
                "time_step": self.step_penalty,
                "rotation_weight": self.w_rotation,
                "progress_multiplier": self.w_progress
            }
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

    def close(self):
        """Ensure all resources are freed when the environment is closed."""
        if self.sim is not None:
            if hasattr(self.sim, 'close'):
                self.sim.close()
    
        plt.close('all')
        
    
        super().close()

    def is_valid_particle(self, x, y):
        return (not self.is_outside_map(x, y))

    def render(self):
            self.sim.render()
            self.sim.reset_plot()
            print(self.state)
            print(len(self.current_path))
            # --- A* path ---
            if self.state == "FOLLOWING" and len(self.current_path) > 0:
                print(f"DEBUG: Rendering Path. Path length: {len(self.current_path)}, Waypoint: {self.waypoint_index}")
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

    def is_valid_pos(self, x, y, robot_radius=0.2):
            """ Checks if the position is valid and clear of obstacles within a radius. """
            res = self.map_meta['resolution']
            gx = int(x / res)
            gy = int(y / res)
            
            # Calculate how many grid cells represent the robot's physical size
            # We use a slightly larger radius (0.3 instead of 0.2) as a safety buffer
            grid_margin = int(robot_radius / res)

            # Define the bounds of the "inflation" square to check
            y_min, y_max = max(0, gy - grid_margin), min(self.occupancy_grid.shape[0], gy + grid_margin + 1)
            x_min, x_max = max(0, gx - grid_margin), min(self.occupancy_grid.shape[1], gx + grid_margin + 1)

            # Check the entire patch. If any pixel is 1 (obstacle), the whole area is invalid.
            patch = self.occupancy_grid[y_min:y_max, x_min:x_max]
            
            # If the patch is empty (out of bounds) or contains any obstacles, return False
            if patch.size == 0 or np.any(patch == 1):
                return False
                
            return True

    def is_outside_map(self, x, y):
        """Keeping this for legacy calls, but it now uses the better grid check"""
        return not self.is_valid_pos(x, y)

    # ==========================================
    # UPDATED RANDOM SPAWN
    # ==========================================
    def _get_random_valid_pos(self, margin=0.3):
        """Uses the occupancy grid to find a truly navigable spawn point."""
        map_obj = self.sim.get_map()
        w, h = map_obj.width, map_obj.height
        
        max_attempts = 100
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
                                
                return x, y
        
        logging.warning("Failed to find valid pos, returning center")
        return w/2, h/2
                
            
            

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
        # Calculate current target error (Ground Truth vs Estimate)
        current_estimate = self.estimated_goal if self.pf_active else self.estimated_goal_ls
        real_goal = self.sim.robot.goal if hasattr(self.sim, 'robot') else self.sim.get_robot_info(0).goal
        target_error = np.linalg.norm(current_estimate - real_goal[:2, 0])

        avg_velocity = self.total_velocity / self.time if self.time > 0 else 0
        
        return {
            'success': terminal and reward > 0,
            'episode_reward': reward, 
            'target_error': target_error, # Key for Graph 2
            'total_distance': self.total_distance,
            'steps': self.time # Key for Graph 4
        }

    def _extract_sim_data(self, action):
        scan = self.sim.get_lidar_scan()
        latest_scan = scan["ranges"] if isinstance(scan, dict) else scan
        robot_state = self.sim.get_robot_state()  
        
        # GROUND TRUTH (For the reward function only)
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
        
        # Navigation math based on the Ghost
        pose_vector = [np.cos(robot_state[2,0]), np.sin(robot_state[2,0])]
        norm_pose = pose_vector / (np.linalg.norm(pose_vector) + 1e-6)
        norm_ghost = goal_vector / (distance_to_ghost + 1e-6)
        
        cos_val = np.dot(norm_pose, norm_ghost)
        sin_val = np.cross(norm_pose, norm_ghost)
        diff_rad = np.arctan2(sin_val, cos_val)
        
        # We return both the Ghost distance (for the robot to act) 
        # and the Real Goal (for the reward to judge)
        return latest_scan, distance_to_ghost, cos_val, sin_val, collision, arrive, diff_rad, action, real_goal

    def reward(self, data):
        latest_scan, dist_ghost, cos, sin, collision, arrive, diff_rad, action, real_goal = data
        total_rew = 0.0
        done = False

        current_estimate = self.estimated_goal if self.pf_active else self.estimated_goal_ls
        
        # Constants 
        lam = 0.01
        d_max = self.set_max_range # e.g., 5.0
        d_min = 0.5                # Minimum distance to avoid collision
        
        # 1. Equation (1): Estimation Error Reward (re)
        est_error = np.linalg.norm(current_estimate - real_goal)
        if est_error > self.rew_err_th:
            re = lam * (0.5 - est_error)
        else:
            re = 1.0

        dist_norm = dist_ghost / self.set_max_range
            
        # 2. Equation (2): Distance Reward (rd)
        if dist_ghost > self.rew_dis_th:
            rd = lam * (0.5 - dist_norm)
        else:
            rd = 1.0

        # 3. Equation (3): Terminal Reward (r_terminal)
        r_terminal = 0
        if dist_ghost > d_max:
            r_terminal = self.collision_penalty  # Out of range penalty
            
        elif dist_ghost < d_min:
            r_terminal = -1
            # Optional: done = True to stop on 'crash'

        #Added

        if self.state == "SEARCHING":
            re_final = re * 15  # Boost estimation signal
            rd_final = rd * 5   # Less aggressive distance reward
        else: # FOLLOWING
            re_final = re * 5   # Reduce estimation focus
            rd_final = rd * 15  # Aggressive path-following focus

        # Reward components
        dir_progress = np.cos(diff_rad) * 0.5 * self.w_progress  # Reward for facing the right direction
        time_penalty = self.step_penalty  # Penalty for each time step
        rotation_penalty = -abs(action[1]) * 0.4  * self.w_rotation # Penalty for excessive rotation

        # Obstacle avoidance
        safe_distance = self.safe_dist
        min_dist = min(latest_scan)
        obstacle_penalty = -(safe_distance - min_dist) if min_dist < safe_distance else 0

        # Final Reward Calculation: r = rd + re + r_terminal + progress_reward + dir_progress + time_penalty + rotation_penalty 
        total_rew = rd_final + r_terminal + re_final + dir_progress + time_penalty + rotation_penalty + obstacle_penalty

        # Also PENALIZE collisions with walls
        if collision:
            total_rew = self.collision_penalty
            done = True

            
        return (latest_scan, dist_ghost, cos, sin, collision, arrive, diff_rad, action, total_rew), done

    def prepare_state(self, data):
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        scan_arr = np.array(latest_scan)
        scan_arr[np.isinf(scan_arr)] = 10

        max_bins = self.state_dim - 7
        bin_size = len(scan_arr) // max_bins
        min_values = [np.min(scan_arr[i:i+bin_size])/self.set_max_range for i in range(0, max_bins * bin_size, bin_size)]

        state = min_values + [distance/self.set_max_range, cos, sin, (action[0]+0.6)/1.2, (action[1]+1.2)/2.4, np.cos(diff_rad), np.sin(diff_rad)]
        return np.array(state, dtype=np.float32), (collision or goal)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if seed is not None:
            # First time reset or manual reset
            random.seed(seed)
            np.random.seed(seed)
        else:
            # Auto-reset: derive a new seed from the internal generator
            # to keep the global libraries (PF, etc.) moving forward deterministically
            new_seed = int(self.np_random.integers(0, 2**31 - 1))
            random.seed(new_seed)
            np.random.seed(new_seed)

        if hasattr(self.sim, 'set_random_seed'):
            self.sim.set_random_seed(seed)

        random_yaml = self.np_random.choice(self.map_files)
        self._load_map(random_yaml, grid_dir=os.path.join(os.path.dirname(MAP_DIR), "OccupancyGrids"))

        self.sim.reset()

        # 1. Randomize Robot Position
        rx, ry = self._get_random_valid_pos(margin=0.5)
        new_theta = self.np_random.uniform(-np.pi, np.pi)
        


        # 2. Randomize Goal Position
        # Ensure goal is not too close to the robot (e.g., at least 3 units away)
        while True:
            gx, gy = self._get_random_valid_pos(margin=0.5)
            dist = np.sqrt((rx - gx)**2 + (ry - gy)**2)
            if dist > 1.0:
                break

        # Update the robot in the simulator
        # self.sim.robot is the internal irsim object
        self.sim.robot.set_state(np.array([[rx], [ry], [new_theta]]))
                
        self.sim.robot.set_goal(np.array([[gx], [gy], [0.0]]))

        # Get the real robot and goal states
        robot_state = self.sim.get_robot_state()
        real_goal = self.sim.robot.goal if hasattr(self.sim, 'robot') else self.sim.get_robot_info(0).goal
        
        # Format robot position for the PF: [x, vx, y, vy]
        robot_pos_pf = [robot_state[0,0], 0.0, robot_state[1,0], 0.0]
        
        # Simulate a range measurement to the target
        initial_dist = np.linalg.norm(robot_state[:2, 0] - real_goal[:2, 0])
        
        
        if self.pf_active:
            # Initialize Particle Filter
            self.pf.init_particles(position=robot_pos_pf, slantrange=initial_dist)
            self.target_tracker.updatePF(dt=0.1, new_range=True, z=initial_dist, myobserver=robot_pos_pf)

            # Then update ghost position from the tracker's result
            self.estimated_goal = np.array([self.target_tracker.pfxs[0], self.target_tracker.pfxs[2]])
        else:
            self.target_tracker.allz = []
            self.target_tracker.eastingpoints_LS = []
            self.target_tracker.northingpoints_LS = []
            self.target_tracker.lsxs = []

            # Update LS
            self.target_tracker.updateLS(dt=0.1, new_range=True, z=initial_dist, myobserver=robot_pos_pf)
            if len(self.target_tracker.lsxs) > 0:
                self.estimated_goal_ls = np.array([self.target_tracker.lsxs[-1][0], self.target_tracker.lsxs[-1][2]])
            
        self.p_pos_origin = robot_state[:2, 0].copy()

        self.state = "SEARCHING"
        self.current_path = []
        self.waypoint_index = 0
        self.nav_goal = self.estimated_goal.copy() if self.pf_active else self.estimated_goal_ls.copy()

        sim_data = self._extract_sim_data(action=[0.0, 0.0])
        sim_data_tuple, done_flag = self.reward(sim_data) 
        obs, terminal = self.prepare_state(sim_data_tuple) 
        
        self._reset_episode_tracking()
        return obs, {}

    def step(self, action):
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
                # GATING LOGIC & STATE MACHINE
                # ==========================================
                if self.state == "SEARCHING":
                    x_var = np.var(self.pf.x[:, 0])
                    y_var = np.var(self.pf.x[:, 2])
                    uncertainty = x_var + y_var
                    
                    self.nav_goal = self.estimated_goal.copy()

                    # Add a cooldown attribute in __init__ (e.g., self.astar_cooldown = 0)
                    if hasattr(self, 'astar_cooldown') and self.astar_cooldown > 0:
                        self.astar_cooldown -= 1

                    elif uncertainty < self.uncertainty_threshold:
                        # FIX 1: Validate that the estimated goal is not inside a wall!
                        if self.is_valid_pos(self.estimated_goal[0], self.estimated_goal[1]):
                            
                            # FIX 2: Do NOT call self.sim.step(stop_action) here. 
                            # The RL step has already been taken at the top of the function.
                            
                            path = self.get_astar_path(robot_state[:2, 0], self.estimated_goal)
                            
                            if path is not None and len(path) > 1:
                                self.current_path = path
                                self.waypoint_index = 0
                                self.state = "FOLLOWING"
                            else:
                                print("A* Failed to find a path. Backing off.")
                                self.state = "SEARCHING"
                                self.current_path = []
                                # FIX 3: Apply a cooldown so it doesn't freeze the game every single step
                                self.astar_cooldown = 10 
                        else:
                            # Estimate is inside a wall, keep searching
                            self.state = "SEARCHING"

                if self.state == "FOLLOWING":
                    # 1. Check if the target drifted heavily from our path destination
                    dist_to_path_end = np.linalg.norm(self.estimated_goal - self.current_path[-1])
                    if dist_to_path_end > 1.5:
                        print("Target drifted! Reverting to SEARCHING.")
                        self.state = "SEARCHING"
                        self.current_path = []
                        self.nav_goal = self.estimated_goal.copy()
                    else:
                        # 2. Update Waypoint Logic
                        target_wp = self.current_path[self.waypoint_index]
                        dist_to_wp = np.linalg.norm(robot_state[:2, 0] - target_wp)

                        # If close to current waypoint, advance to the next one
                        if dist_to_wp < 0.4 and self.waypoint_index < len(self.current_path) - 1:
                            self.waypoint_index += 1
                            target_wp = self.current_path[self.waypoint_index]

                        # 3. Set the nav_goal for the RL Agent to the active waypoint
                        self.nav_goal = np.array(target_wp)
                # ==========================================

            else: # LS MODE
                self.target_tracker.updateLS(dt=0.1, new_range=True, z=dist_z, myobserver=robot_pos_pf)
                
                if len(self.target_tracker.lsxs) > 0:
                    # Current raw LS estimate
                    new_ls_estimate = np.array([self.target_tracker.lsxs[-1][0], self.target_tracker.lsxs[-1][2]])
                    
                    if self.state == "SEARCHING":
                        # LS Confidence Metric: Need at least 4-5 unique points for a stable trilateration
                        num_points = len(self.target_tracker.eastingpoints_LS)
                        
                        # Update the goal for the RL agent
                        self.estimated_goal_ls = new_ls_estimate
                        self.nav_goal = self.estimated_goal_ls.copy()

                        # Gating: Switch to FOLLOWING if we have enough data and it's not in a wall
                        if num_points >= 5: 
                            if self.is_valid_pos(new_ls_estimate[0], new_ls_estimate[1]):
                                stop_action = np.array([[0.0], [0.0]])
                                self.sim.step(stop_action)

                                path = self.get_astar_path(robot_state[:2, 0], new_ls_estimate)
                                if path is not None and len(path) > 1:
                                    self.current_path = path
                                    self.waypoint_index = 0
                                    self.state = "FOLLOWING"
                                    self.estimated_goal_ls = new_ls_estimate
                    
                    elif self.state == "FOLLOWING":
                        # 1. Check for LS "Jumps" (Instability)
                        # If the LS solver suddenly moves the goal > 1.0m, the path is invalid
                        ls_drift = np.linalg.norm(new_ls_estimate - self.estimated_goal_ls)
                        dist_to_path_end = np.linalg.norm(new_ls_estimate - self.current_path[-1])

                        if ls_drift > 1.0 or dist_to_path_end > 1.5:
                            print("LS Estimate Jumped! Reverting to SEARCHING.")
                            self.state = "SEARCHING"
                            self.current_path = []
                            self.estimated_goal_ls = new_ls_estimate
                            self.nav_goal = self.estimated_goal_ls.copy()
                        else:
                            # 2. Waypoint Following logic (Same as PF)
                            target_wp = self.current_path[self.waypoint_index]
                            dist_to_wp = np.linalg.norm(robot_state[:2, 0] - target_wp)

                            if dist_to_wp < 0.4 and self.waypoint_index < len(self.current_path) - 1:
                                self.waypoint_index += 1
                                target_wp = self.current_path[self.waypoint_index]

                            self.nav_goal = np.array(target_wp)
                            
                            self.estimated_goal_ls = new_ls_estimate

            # Process normal step data
            sim_data = self._extract_sim_data(action=action)
            sim_data_tuple, done_flag = self.reward(sim_data) 
            obs, terminal = self.prepare_state(sim_data_tuple) 
            reward = sim_data_tuple[-1]

            self._calculate_metrics(robot_state[:2, 0], action)
            self.time += 1
            
            terminated = bool(terminal)
            truncated = self.time >= self.max_steps
            
            if truncated and not terminated:
                reward = -100  # Penalize for timeout if not already terminated

            if self.render_mode:
                self.render() 

            return obs, float(reward), terminated, truncated, self._get_episode_info(terminated, reward)