from importlib import resources
from pathlib import Path
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward, WayPointReward, PointReward
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

        if render_mode is not None:

            with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
                self.model_path = str(asset_path)
            self.renderer = BlueRovRenderer()
            self.render_mode = render_mode

        self.target_point = [0.0, -1.0, -5.0, 0.0]  # x, y, z, theta
        
        init_x = 0
        init_y = 0
        init_z = -5.0
        init_theta = 0
        self.start_point = [init_x, init_y, init_z, init_theta]

        self.reward_fn = PointReward(self.target_point,threshold=0.01)

        self.dynamics = Dynamics()

        # Initialize state variables
        self.state = {
            "x": init_x,  # x position (m)
            "y": init_y,  # y position (m)
            "z": init_z,  # depth (m)
            "theta": init_theta,  # heading angle (rad)
            "vx": 0,  # x velocity (m/s)
            "vy": 0,  # y velocity (m/s)
            "vz": 0,  # vertical velocity (m/s)
            "omega": 0,  # angular velocity (rad/s)
        }

        self.init_state = deepcopy(self.state)

        print(
            "Initial state for env: ",
            self.init_state["x"],
            self.init_state["y"],
            self.init_state["z"],
            self.init_state["theta"],
        )

        # Define action space: 4 normalized thruster commands between -1.0 and 1.0
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # the state space is only partially observable, we will only provide the offset to the current waypoint
        self.observation_space = spaces.Dict(
            {
                "x_offset": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "y_offset": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "z_offset": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "theta_offset": spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Simulation parameters
        self.dt = 0.1  # Time step (seconds)
        self.render_mode = render_mode
        self.trajectory_file = trajectory_file

        self.action_steps = 0

        if self.render_mode == "human":
            self.render()

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

        self.state = deepcopy(self.init_state)

        self.disturbance_dist = self.dynamics.reset()

        # Convert dictionary values to numpy arrays for the observation
        obs = {
            "x_offset": np.array([0], dtype=np.float32),
            "y_offset": np.array([0], dtype=np.float32),
            "z_offset": np.array([0], dtype=np.float32),
            "theta_offset": np.array([0], dtype=np.float32),
        }

        self.action_steps = 0

        if self.render_mode is not None:
            self.step_sim()

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
        obs = {}

        self.action_steps += 1
        print("action_steps: ", self.action_steps)

        # Add offsets to the observation
        waypoint = self.target_point
        obs["x_offset"] = np.array(
            [waypoint[0] - self.state["x"]], dtype=np.float32
        )
        obs["y_offset"] = np.array(
            [waypoint[1] - self.state["y"]], dtype=np.float32
        )
        obs["z_offset"] = np.array(
            [waypoint[2] - self.state["z"]], dtype=np.float32
        )
        obs["theta_offset"] = np.array(
            [waypoint[3] - self.state["theta"]], dtype=np.float32
        )

        # Calculate reward based on current state
        reward = self.reward_fn.get_reward(self.state)

        # Determine if episode should terminate
        terminated = False

        # Check boundary conditions for termination
        if self.state["z"] < -10.0:  # Depth limit
            terminated = True
        if self.state["z"] > 0.1:  # Depth limit
            terminated = True
        if (
            abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0
        ):  # Horizontal boundaries
            terminated = True

        truncated = False  # Episode is not truncated

        info = {
        }

        if self.render_mode == "human":
            self.step_sim()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment if in human mode.
        """
        self.renderer.render(self.model_path, self.init_state)

        self.renderer.visualize_waypoints(
            [self.start_point[:3],self.target_point[:3]],
            current_idx=1,
        )

    def step_sim(self):
        """
        Update the visualization with the current state.
        """
        if self.state == self.init_state:
            pass
            # print("i was sent home")
        self.renderer.step_sim(self.state)
        self.renderer.visualize_waypoints(
            [self.start_point[:3],self.target_point[:3]],
            current_idx=1,
        )
