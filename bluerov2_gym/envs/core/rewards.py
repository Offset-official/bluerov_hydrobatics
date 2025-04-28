import numpy as np
import csv
from pathlib import Path


class Reward:
    def __init__(self):
        pass

    def get_reward(self, obs):
        position_error = np.sqrt(obs["x"][0] ** 2 + obs["y"][0] ** 2 + obs["z"][0] ** 2)

        # Velocity penalty
        velocity_penalty = np.sqrt(
            obs["vx"][0] ** 2 + obs["vy"][0] ** 2 + obs["vz"][0] ** 2
        )

        # Orientation error
        orientation_error = abs(obs["theta"][0])

        # Combined reward
        reward = -(
            1.0 * position_error  # Weight for position error
            + 0.1 * velocity_penalty  # Weight for velocity
            + 0.5 * orientation_error  # Weight for orientation
        )

        return reward


class SinglePointReward:
    def __init__(self, threshold=0.1):
        """
        Initialize the SinglePointReward class.
        goal_point: numpy array of shape (4,) containing x, y, z coordinates and heading angle (rad)
        """
        self.threshold = threshold

    def get_reward(self, distance_to_goal, theta_offset, action_magnitude):
        r_completion = 0
        if distance_to_goal < self.threshold:
            distance_to_goal = 0.0
            r_completion = 100

        r_pos = np.exp(-(distance_to_goal**2))
        r_angle = 0.1 * np.exp(-(theta_offset**2))
        r_action = 0.05 * np.exp(-(action_magnitude))

        return np.array([r_pos, r_angle, r_action, r_completion])


class WayPointReward:
    def __init__(self, waypoints, threshold=0.1):
        """
        Initialize the WayPointReward class.
        waypoints: numpy array of shape (num_points, 4) containing x, y, z, coordinates and heading angle (rad)
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.threshold = threshold
        self.total_waypoints = len(waypoints)

    def get_reward(self, obs):
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        current_waypoint = self.waypoints[self.current_waypoint_idx]
        position_error = np.sqrt(
            (obs["x"][0] - current_waypoint[0]) ** 2
            + (obs["y"][0] - current_waypoint[1]) ** 2
            + (obs["z"][0] - current_waypoint[2]) ** 2
        )

        orientation_error = abs(obs["theta"][0] - current_waypoint[3])

        # Check if the waypoint is reached
        if position_error < self.threshold:
            self.current_waypoint_idx += 1

        return -(position_error + orientation_error)
