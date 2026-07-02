import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Custom2DDroneEnv(gym.Env):
    """
    A minimal custom 2D Drone Environment for Gymnasium.
    State: [x, z, vx, vz, theta, theta_dot]
    Actions: [thrust_left, thrust_right] in [0, 1]
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, mode='explore', max_steps=1000):
        super().__init__()
        self.mode = mode
        self.max_steps = max_steps
        
        # Physics constants
        self.mass = 1.0
        self.gravity = 9.81
        self.arm_length = 0.2
        self.inertia = 0.05
        
        # We want hover to be around action = 0.5 for both motors.
        # Total force at hover = mass * gravity = 9.81
        # So each motor at 0.5 thrust should produce 9.81 / 2 = 4.905 force.
        # max_thrust for one motor = 9.81 / 2 / 0.5 = 9.81
        # Total max thrust = 19.62 (2g)
        self.max_thrust = self.mass * self.gravity
        self.dt = 0.02
        
        # Bounds for the observation space
        high = np.array([
            10.0, # x
            10.0, # z
            20.0, # vx
            20.0, # vz
            np.pi, # theta
            10.0  # theta_dot
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # Action space: thrust left, thrust right in [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.state = None
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Start near the ground, upright
        # Format: [x, z, vx, vz, theta, theta_dot]
        # Adding slight noise to initial conditions
        x0 = self.np_random.uniform(-0.1, 0.1)
        z0 = self.np_random.uniform(0.1, 0.2)
        vx0 = self.np_random.uniform(-0.05, 0.05)
        vz0 = self.np_random.uniform(-0.05, 0.05)
        theta0 = self.np_random.uniform(-0.05, 0.05)
        theta_dot0 = self.np_random.uniform(-0.05, 0.05)
        
        self.state = np.array([x0, z0, vx0, vz0, theta0, theta_dot0], dtype=np.float32)
        return self.state, {}
        
    def step(self, action):
        self.step_count += 1
        
        x, z, vx, vz, theta, theta_dot = self.state
        tl, tr = np.clip(action, 0.0, 1.0)
        
        # Forces
        thrust_l = tl * self.max_thrust
        thrust_r = tr * self.max_thrust
        total_thrust = thrust_l + thrust_r
        
        # Accelerations
        ax = -total_thrust * np.sin(theta) / self.mass
        az = (total_thrust * np.cos(theta) / self.mass) - self.gravity
        
        # Torques
        torque = (thrust_r - thrust_l) * self.arm_length
        alpha = torque / self.inertia
        
        # Euler integration
        vx_new = vx + ax * self.dt
        vz_new = vz + az * self.dt
        theta_dot_new = theta_dot + alpha * self.dt
        
        x_new = x + vx_new * self.dt
        z_new = z + vz_new * self.dt
        theta_new = theta + theta_dot_new * self.dt
        
        # Normalize theta to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        
        self.state = np.array([x_new, z_new, vx_new, vz_new, theta_new, theta_dot_new], dtype=np.float32)
        
        # Termination conditions (Crash)
        crashed = z_new < 0.0 or abs(theta_new) > (np.pi / 2.0)
        out_of_bounds = abs(x_new) > 10.0 or z_new > 10.0
        
        terminated = bool(crashed or out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)
        
        reward = 0.0
        if self.mode == 'hover':
            # Target is x=0, z=1.0
            dist_penalty = 0.1 * (x_new**2 + (z_new - 1.0)**2)
            angle_penalty = 0.1 * (theta_new**2 + 0.1 * theta_dot_new**2)
            action_penalty = 0.01 * (tl**2 + tr**2)
            
            reward = 1.0 - dist_penalty - angle_penalty - action_penalty
            if crashed:
                reward -= 10.0
                
        elif self.mode == 'explore':
            # Handled externally by curiosity wrapper, but baseline survival uses +1
            reward = 1.0
            if crashed:
                reward -= 10.0
                
        info = {
            'crashed': crashed,
            'out_of_bounds': out_of_bounds
        }
        
        return self.state, float(reward), terminated, truncated, info
