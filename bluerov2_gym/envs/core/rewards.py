import numpy as np


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
    def __init__(self, threshold=0.25, angular_threshold=0.1):
        """
        Initialize the SinglePointReward class.
        goal_point: numpy array of shape (4,) containing x, y, z coordinates and heading angle (rad)
        """
        self.threshold = threshold
        self.angular_threshold = angular_threshold

    import numpy as np

    def wrap_to_pi(self, angle):
        """Wrap any angle in radians to [−π, +π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_reward(
            self,
            distance_to_goal,
            theta,
            target_theta,
            action_magnitude,
            number_of_steps,
            dot_to_goal=0.0,
            last_distance_to_goal=0.0,
            offset_x=0.0,
            offset_y=0.0,
            offset_z=0.0,
            offset_x_last=0.0,
            offset_y_last=0.0,
            offset_z_last=0.0,
            last_closest_distance_to_goal=0.0,
            terminated=False,
        ):
        """
        Reward that
        • requires you to face the waypoint to get the big completion bonus,
        • adds a cosine‐based heading alignment bonus,
        • only rewards forward‐velocity when heading is within a small cone,
        • does NOT reward yaw‐actions (so turning is “free” of penalty),
        • keeps your previous distance‐based shaping, step‐penalty, and dot‐to‐goal.
        """

        # 1) Completion bonus only if both position AND heading are within threshold
        r_completion = 0
        heading_error = self.wrap_to_pi(theta - target_theta)
        if (distance_to_goal < self.threshold
            and abs(heading_error) < self.angular_threshold):
            distance_to_goal = 0.0
            r_completion = 1500

        # 2) Position‐based shaping (as before)
        pos_reward = np.exp(-(distance_to_goal**3))

        # 3) Angle‐to‐target‐heading penalty (unchanged)
        angle_reward = -abs(heading_error)

        # 4) Progress‐toward‐goal shaping (unchanged)
        progress_reward = 15.0 * (last_closest_distance_to_goal - distance_to_goal)

        # 5) Per‐step time penalty (to discourage loitering)
        time_penalty = 0.05 * number_of_steps

        # 6) Action penalty: only on translational magnitude, not on yaw‑torque
        #    so the agent can turn “for free” to fix heading.
        #    We assume action_magnitude is a vector [ax, ay, az, ayaw];
        #    if it’s a scalar sum you may need to split out yaw component.
        act = action_magnitude
    # detect if iterable (vector) or scalar:
        try:
            # if it has length ≥3, treat first 3 as translation
            if len(act) >= 3:
                trans_action_mag = np.linalg.norm(np.array(act[:3]))
            else:
                # vector but <3 dims? just norm the whole thing
                trans_action_mag = np.linalg.norm(np.array(act))
        except TypeError:
            # not iterable → scalar
            trans_action_mag = float(act)
            action_penalty = np.exp(-(trans_action_mag**2))

        # 7) Cosine‐based heading alignment bonus
        w_heading = 3.0
        heading_bonus = w_heading * np.cos(heading_error)

        # 8) Gate forward‐velocity reward on good heading
        w_fwd = 0.1
        gate_thresh = 15 * np.pi / 180.0  # 15°
        fwd_reward = 0.0
        if abs(heading_error) < gate_thresh:
            fwd_reward = w_fwd * dot_to_goal

        # 9) Penalty on termination by timeout/collision
        term_penalty = -500.0 if terminated else 0.0

        # Sum all terms
        total_reward = (
            r_completion
            + pos_reward
            + angle_reward
            + progress_reward
            - time_penalty
            + action_penalty
            + heading_bonus
            + fwd_reward
            + term_penalty
        )

        # If you need to log individual terms, you can return them in reward_tuple
        reward_tuple = np.array([
            pos_reward,
            angle_reward,
            progress_reward,
            -time_penalty,
            action_penalty,
            heading_bonus,
            fwd_reward,
            r_completion,
            term_penalty
        ])

        return (total_reward, reward_tuple)


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
