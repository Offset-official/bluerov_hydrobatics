from importlib import resources
from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward, SinglePointReward
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

    def __init__(self, render_mode=None):
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

        self.dynamics = Dynamics()

        self.state = {
            "x": 0.0,  # x position (m)
            "y": 0.0,  # y position (m)
            "z": 0.0,  # depth (m)
            "theta": 0.0,  # heading angle (rad)
            "vx": 0.0,  # x velocity (m/s)
            "vy": 0.0,  # y velocity (m/s)
            "vz": 0.0,  # vertical velocity (m/s)
            "omega": 0.0,  # angular velocity (rad/s)
        }

        self.init_state = deepcopy(self.state)

        self.goal_point = np.array([0, -1, 0, 0], dtype=np.float64)  # x,y,z,theta (yaw)

        self.threshold_distance = 0.1

        self.reward_fn = SinglePointReward(threshold=self.threshold_distance)

        # 4 normalized thruster commands between -1.0 and 1.0
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "offset_x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "offset_y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "offset_z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "offset_theta": spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float64
                ),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )

        self.dt = 0.1  # Time step (seconds)
        self.render_mode = render_mode

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

        obs = self.compute_observation()

        info = {
            "distance_from_goal": self.compute_distance_from_goal(),
        }

        return obs, info

    def step(self, action):
        """
        Take a step in the environment given an action.

        Args:
            action (numpy.ndarray): Array of 4 thruster commands (-1.0 to 1.0)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.dynamics.step(self.state, action)

        obs = self.compute_observation()

        # Reset conditions
        terminated = False
        truncated = False

        # Check boundary conditions for termination
        if abs(self.state["z"]) > 10.0:  # Depth limit
            terminated = True
        if (
            abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0
        ):  # Horizontal boundaries
            terminated = True

        distance_from_goal = self.compute_distance_from_goal()

        is_success = bool(distance_from_goal < self.threshold_distance)
        terminated = bool(terminated or is_success)

        reward = self.reward_fn.get_reward(distance_from_goal, obs["offset_theta"][0])

        info = {
            "distance_from_goal": distance_from_goal,
            "reward": reward,
            "is_success": is_success,
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
            [[0, 0, 0], self.goal_point[:3]],
            current_idx=1,
        )

    def step_sim(self):
        """
        Update the visualization with the current state.
        """
        self.renderer.step_sim(self.state)
        self.renderer.visualize_waypoints(
            [[0, 0, 0], self.goal_point[:3]],
            current_idx=1,
        )

    def compute_observation(self):

        obs = {
            "offset_x": np.array([self.state["x"] - self.goal_point[0]]),
            "offset_y": np.array([self.state["y"] - self.goal_point[1]]),
            "offset_z": np.array([self.state["z"] - self.goal_point[2]]),
            "offset_theta": np.array([self.state["theta"] - self.goal_point[3]]),
            "vx": np.array([self.state["vx"]]),
            "vy": np.array([self.state["vy"]]),
            "vz": np.array([self.state["vz"]]),
            "omega": np.array([self.state["omega"]]),
        }

        return obs

    def compute_distance_from_goal(self):

        return np.linalg.norm(
            np.array(
                [
                    self.state["x"] - self.goal_point[0],
                    self.state["y"] - self.goal_point[1],
                    self.state["z"] - self.goal_point[2],
                ]
            )
        )
