from importlib import resources

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer


class BlueRov(gym.Env):
    """
    BlueROV2 Gymnasium Environment
    
    This environment simulates the dynamics of a BlueROV2 underwater vehicle
    for reinforcement learning tasks. It includes position and velocity states
    in a 3D environment with heading angle.
    
    State variables:
    - x, y, z: 3D position coordinates
    - theta: heading angle
    - vx, vy, vz: linear velocities
    - omega: angular velocity
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_file=None):
        """
        Initialize the BlueROV environment
        
        Args:
            render_mode (str, optional): Rendering mode. Use "human" for visualization.
        """
        super().__init__()
        
        # Load 3D model for visualization
        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)
            
        # Initialize renderer if human rendering is requested
        if render_mode == "human":
            self.renderer = BlueRovRenderer()
            
        # Initialize reward function and dynamics model
        self.reward_fn = Reward()
        self.dynamics = Dynamics()
        
        # Initialize state variables
        self.state = {
            "x": 0,      # x position (m)
            "y": 0,      # y position (m)
            "z": 0,      # depth (m)
            "theta": 0,  # heading angle (rad)
            "vx": 0,     # x velocity (m/s)
            "vy": 0,     # y velocity (m/s)
            "vz": 0,     # vertical velocity (m/s)
            "omega": 0,  # angular velocity (rad/s)
        }

        # Define action space: 4 normalized thruster commands between -1.0 and 1.0
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Define observation space: Dictionary of all state variables
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "theta": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        
        # Simulation parameters
        self.dt = 0.1  # Time step (seconds)
        self.render_mode = render_mode
        self.trajectory_file = trajectory_file

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional configuration options
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        # Reset state to initial conditions (at origin with zero velocity)
        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "theta": 0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "omega": 0,
        }

        # Reset the dynamics model and get distribution of possible disturbances
        self.disturbance_dist = self.dynamics.reset()
        
        # Convert dictionary values to numpy arrays for the observation
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

        return obs, {}

    def step(self, action):
        """
        Take a step in the environment given an action.
        
        Args:
            action (numpy.ndarray): Array of 4 thruster commands (-1.0 to 1.0)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Update state according to dynamics model
        self.dynamics.step(self.state, action)
        
        # Format observation as required by Gymnasium
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

        # Calculate reward based on current state
        if self.trajectory_file is not None:
            reward = self.reward_fn.get_reward_trajectory(obs, action, self.trajectory_file)
        else:
            reward = self.reward_fn.get_reward(obs)

        # Determine if episode should terminate
        terminated = False
        
        # Check boundary conditions for termination
        if abs(self.state["z"]) > 10.0:  # Depth limit
            terminated = True
        if abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0:  # Horizontal boundaries
            terminated = True

        truncated = False  # Episode is not truncated

        return obs, reward, terminated, truncated, {}

    def render(self):
        """
        Render the environment if in human mode.
        """
        self.renderer.render(self.model_path)

    def step_sim(self):
        """
        Update the visualization with the current state.
        """
        self.renderer.step_sim(self.state)


    def set_waypoints_visualization(self, waypoints):
        """
        Set the trajectory waypoints for visualization.
        
        Args:
            waypoints (numpy.ndarray): Array of shape (num_points, 3) with x, y, z coordinates
        """
        if self.renderer:
            self.renderer.set_waypoints(waypoints)
        else:
            raise RuntimeError("Renderer not initialized. Set render_mode to 'human'.")