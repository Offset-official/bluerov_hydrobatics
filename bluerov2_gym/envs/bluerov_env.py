from importlib import resources
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward, WaypointReward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        trajectory_path=None,
        waypoint_reward=False,
        waypoint_threshold=0.5,
    ):
        super().__init__()
        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.renderer = BlueRovRenderer()

        self.waypoint_reward = waypoint_reward
        if waypoint_reward and trajectory_path:
            self.reward_fn = WaypointReward(
                trajectory_path, threshold=waypoint_threshold
            )
        else:
            self.reward_fn = Reward()

        self.dynamics = Dynamics()
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

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

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
        self.dt = 0.1  # Time step
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

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

        if self.waypoint_reward:
            self.reward_fn.reset()

        self.disturbance_dist = self.dynamics.reset()
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

        return obs, {}

    def step(self, action):
        self.dynamics.step(self.state, action)
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

        reward = self.reward_fn.get_reward(obs)

        terminated = False
        # Example conditions (please change these to your own conditions)
        if abs(self.state["z"]) > 10.0:
            terminated = True
        if abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0:
            terminated = True

        if self.waypoint_reward and hasattr(self.reward_fn, "current_waypoint_idx"):
            if self.reward_fn.current_waypoint_idx >= self.reward_fn.total_waypoints:
                terminated = True
                print(
                    f"Trajectory completed! Reached {self.reward_fn.reached_waypoints}/{self.reward_fn.total_waypoints} waypoints"
                )

        truncated = False

        info = {
            "waypoint_progress": (
                self.reward_fn.current_waypoint_idx / self.reward_fn.total_waypoints
                if self.waypoint_reward
                else 0.0
            )
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        self.renderer.render(self.model_path)

        if self.waypoint_reward and hasattr(self.reward_fn, "trajectory"):
            self.renderer.visualize_waypoints(
                self.reward_fn.trajectory,
                current_idx=self.reward_fn.current_waypoint_idx,
            )

    def step_sim(self):
        self.renderer.step_sim(self.state)
