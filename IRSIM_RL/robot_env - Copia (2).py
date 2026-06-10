import numpy as np
import gymnasium as gym
from gymnasium import spaces
import irsim
import os
import sys
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Tracking.target_pf import Target


# ==========================================
# Gymnasium Environment definition
# ==========================================



class RobotNavEnv(gym.Env):
    def __init__(self, render=False, world_file="robot_world.yaml", pf_active=False):
        super(RobotNavEnv, self).__init__()
        
        if pf_active:

            self.target_tracker = Target(method='range', max_pf_range=15)
            self.pf = self.target_tracker.pf
            self.particle_plot = None
        else:
            self.target_tracker = Target(method='range')

        self.pf_active = pf_active

        self.rew_err_th = 0.05
        self.rew_dis_th = 1.4
        self.set_max_range = 15
        self.orbit_radius = 0.9
        
        self.render_mode = render
        self.state_dim = 49  
        self.max_steps = 200  
        
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
        
        # Initialize simulator
        self.sim = irsim.make(world_file, display=render)
        if pf_active:
            self.pf.set_validation_callback(self.is_valid_particle)
  
        self._reset_episode_tracking()

    def is_valid_particle(self, x, y):
        return (not self.is_outside_map(x, y))

    def render(self):
            self.sim.render()
            self.sim.reset_plot()

            # --- GHOST AND LS POINTS ---
            if self.pf_active:
                # (Keep your existing PF ghost rendering here)
                if hasattr(self, 'estimated_goal'):
                    gx, gy = self.estimated_goal
                    self.sim.draw_points(points=[[gx, gy]], c='magenta', s=300)
            else:
                # Render the LS Ghost (Red)
                if hasattr(self, 'estimated_goal_ls'):
                    gx, gy = self.estimated_goal_ls
                    self.sim.draw_points(points=[[gx, gy]], c='red', s=300)

                # --- NEW: RENDER LS OBSERVER POINTS ---
                # These are the robot positions used to calculate the estimate
                if hasattr(self.target_tracker, 'eastingpoints_LS'):
                    ex = self.target_tracker.eastingpoints_LS
                    ey = self.target_tracker.northingpoints_LS
                    
                    if len(ex) > 0:
                        # Ensure we are passing float values, not numpy types
                        obs_points = [[float(x), float(y)] for x, y in zip(ex, ey)]
                        self.sim.draw_points(points=obs_points, c='cyan', s=5)
                    

            # --- PARTICLES ---
            if self.pf_active:
                if hasattr(self.pf, 'x') and self.pf.x is not None:
                    px = self.pf.x[:, 0]
                    py = self.pf.x[:, 2]
                    points = np.stack([px, py], axis=1).tolist()
                    self.sim.draw_points(points=points, c='cyan', s=5)

    def is_outside_map(self, x, y):
        map_obj = self.sim.get_map()
      
        # Based on your print: 16.16x13.16
        width_w = map_obj.width
        height_h = map_obj.height

        if x < 1 or x > width_w:
            return True
        if y < 1 or y > height_h:
            return True
            
        return False
    

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

        goal_vector = [self.estimated_goal[0] - robot_state[0,0], self.estimated_goal[1] - robot_state[1,0]]
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

    def reward(self, data): #Aggiungere premio/penalità se non si muove e controllare se le formule sono corrette con i valori del verde
        latest_scan, dist_ghost, cos, sin, collision, arrive, diff_rad, action, real_goal = data
        total_rew = 0.
        done = False
        
        # Constants 
        lam = 0.01
        d_max = self.set_max_range # e.g., 5.0
        d_min = 0.5                # Minimum distance to avoid collision
        
        # 1. Equation (1): Estimation Error Reward (re)
        est_error = np.linalg.norm(self.estimated_goal - real_goal)
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
            r_terminal = -100
            
        elif dist_ghost < d_min:
            r_terminal = -1
            # Optional: done = True to stop on 'crash'

        re_final = re * 10 # Amplify the "Estimation" signal
        rd_final = rd * 10 # Amplify the "Distance" signal

                # Reward components
        dir_progress = np.cos(diff_rad) * 0.5
        time_penalty = -0.1  # Penalty for each time step
        rotation_penalty = -abs(action[1]) * 0.4  # Penalty for excessive rotation

                # Obstacle avoidance
        safe_distance = 0.5
        min_dist = min(latest_scan)
        obstacle_penalty = -(safe_distance - min_dist) if min_dist < safe_distance else 0

        # Final Reward Calculation: r = rd + re + r_terminal + progress_reward + dir_progress + time_penalty + rotation_penalty 
        total_rew = rd_final + r_terminal + re_final + dir_progress + time_penalty + rotation_penalty + obstacle_penalty

        # Also PENALIZE collisions with walls
        if collision:
            total_rew = -100
            done = True



        #if est_error <= self.rew_err_th:
            
        #else:
        #    total_rew = (re_final * rd_final) + r_terminal
            
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
        self.sim.reset()

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

        sim_data = self._extract_sim_data(action=[0.0, 0.0])
        sim_data_tuple, done_flag = self.reward(sim_data) 
        obs, terminal = self.prepare_state(sim_data_tuple) 
        
        self._reset_episode_tracking()
        return obs, {}

    def step(self, action):
        # Move the robot
        ctrl_action = np.array([[action[0]], [action[1]]])
        self.sim.step(ctrl_action)
        
        # --- PARTICLE FILTER ---

        # 2. Get robot state and real measurement
        robot_state = self.sim.get_robot_state()
        robot_pos_pf = np.array([robot_state[0,0], 0.0, robot_state[1,0], 0.0])
        real_goal = self.sim.robot.goal if hasattr(self.sim, 'robot') else self.sim.get_robot_info(0).goal
        dist_z = np.linalg.norm(robot_state[:2, 0] - real_goal[:2, 0])
        
        if self.pf_active:
            self.target_tracker.updatePF(dt=0.1, new_range=True, z=dist_z, myobserver=robot_pos_pf)

            # Then update ghost position from the tracker's result
            self.estimated_goal = np.array([self.target_tracker.pfxs[0], self.target_tracker.pfxs[2]])
        else:
   
            self.target_tracker.updateLS(dt=0.1, new_range=True, z=dist_z, myobserver=robot_pos_pf)
            if len(self.target_tracker.lsxs) > 0:
                # Note: LS needs at least 4 points to start producing valid Plsu values
                self.estimated_goal_ls = np.array([self.target_tracker.lsxs[-1][0], self.target_tracker.lsxs[-1][2]])

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
            reward = -100

        if self.render_mode:
            self.render() # Explicitly call the irsim render

        return obs, float(reward), terminated, truncated, self._get_episode_info(terminated, reward)