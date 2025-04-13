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


class WaypointReward:
    def __init__(self, trajectory_path, threshold=0.5):
        """
        Initialize waypoint-based reward function

        Args:
            trajectory_path: Path to CSV file containing waypoints
            threshold: Distance threshold for considering a waypoint reached (meters)
        """
        self.trajectory = self.load_trajectory(trajectory_path)
        self.current_waypoint_idx = 0
        self.threshold = threshold
        self.reached_waypoints = 0
        self.total_waypoints = len(self.trajectory)

        # Initialize the first waypoint
        if self.trajectory:
            self.current_waypoint = self.trajectory[0]
            print(f"Initial waypoint: {self.current_waypoint}")
        else:
            raise ValueError("No waypoints loaded from trajectory file")

    def load_trajectory(self, file_path):
        """
        Load trajectory waypoints from a CSV file

        Args:
            file_path: Path to the CSV file containing x, y, z coordinates

        Returns:
            List of waypoints as [x, y, z] coordinates
        """
        waypoints = []
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row

            if header != ["x", "y", "z"]:
                print(
                    f"Warning: CSV header {header} doesn't match expected format ['x', 'y', 'z']"
                )

            for row in reader:
                if len(row) >= 3:
                    try:
                        x, y, z = float(row[0]), float(row[1]), float(row[2])
                        waypoints.append([x, y, z])
                    except ValueError:
                        print(f"Warning: Skipping invalid row {row}")

        print(f"Loaded {len(waypoints)} waypoints from {file_path}")
        return waypoints

    def get_reward(self, obs):
        """
        Calculate reward based on distance to current waypoint

        Args:
            obs: Current observation

        Returns:
            reward: Calculated reward value
        """
        if self.current_waypoint_idx >= len(self.trajectory):
            return 0

        waypoint = self.current_waypoint

        distance_error = np.sqrt(
            (obs["x"][0] - waypoint[0]) ** 2
            + (obs["y"][0] - waypoint[1]) ** 2
            + (obs["z"][0] - waypoint[2]) ** 2
        )

        waypoint_reached = distance_error < self.threshold

        if waypoint_reached:
            self.current_waypoint_idx += 1
            self.reached_waypoints += 1

            if self.current_waypoint_idx < len(self.trajectory):
                self.current_waypoint = self.trajectory[self.current_waypoint_idx]
                # print( f"Reached waypoint {self.current_waypoint_idx-1}, moving to next: {self.current_waypoint}")
            else:
                print("All waypoints reached!")

        return -distance_error

    def reset(self):
        """Reset waypoint tracking to the beginning of the trajectory"""
        self.current_waypoint_idx = 0
        self.reached_waypoints = 0
        if self.trajectory:
            self.current_waypoint = self.trajectory[0]
        return
