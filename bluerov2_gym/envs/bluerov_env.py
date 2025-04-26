from importlib import resources
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward, WayPointReward
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

        if trajectory_file is not None:
            self.trajectory = np.loadtxt(trajectory_file, delimiter=",")
            print(f"Loaded trajectory with {self.trajectory.shape[0]} waypoints")
            init_x = self.trajectory[0, 0]
            init_y = self.trajectory[0, 1]
            init_z = self.trajectory[0, 2]
            init_theta = self.trajectory[0, 3]
        else:
            init_x = 0
            init_y = 0
            init_z = 0
            init_theta = 0
            self.trajectory = None

        if self.trajectory is not None:
            self.reward_fn = WayPointReward(self.trajectory)
        else:
            self.reward_fn = Reward()

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

        self.init_state = self.state

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
        print("Reset called")

        self.state = self.init_state

        self.disturbance_dist = self.dynamics.reset()

        # Convert dictionary values to numpy arrays for the observation
        obs = {
            "x_offset": np.array([0], dtype=np.float32),
            "y_offset": np.array([0], dtype=np.float32),
            "z_offset": np.array([0], dtype=np.float32),
            "theta_offset": np.array([0], dtype=np.float32),
        }

        if self.render_mode is not None:
            print("I should go home")
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

        if self.trajectory is not None:
            # Add offsets to the observation
            current_waypoint_idx = self.reward_fn.current_waypoint_idx
            waypoint = self.trajectory[current_waypoint_idx]
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
        else:
            obs["x_offset"] = np.array([0.0], dtype=np.float32)
            obs["y_offset"] = np.array([0.0], dtype=np.float32)
            obs["z_offset"] = np.array([0.0], dtype=np.float32)
            obs["theta_offset"] = np.array([0.0], dtype=np.float32)

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
        if self.trajectory is not None:
            waypoint_progress = (
                self.reward_fn.current_waypoint_idx / self.reward_fn.total_waypoints
            )

        info = {
            "waypoint_progress": (
                waypoint_progress if self.trajectory is not None else 0.0
            )
        }

        if self.render_mode == "human":
            self.step_sim()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment if in human mode.
        """
        self.renderer.render(self.model_path, self.init_state)

        if self.trajectory is not None:
            self.renderer.visualize_waypoints(
                self.trajectory[:, :3],
                current_idx=self.reward_fn.current_waypoint_idx,
            )

    def step_sim(self):
        """
        Update the visualization with the current state.
        """
        self.renderer.step_sim(self.state)
        if self.trajectory is not None:
            self.renderer.visualize_waypoints(
                self.trajectory[:, :3],
                current_idx=self.reward_fn.current_waypoint_idx,
            )
