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
    def __init__(self,
                 threshold: float = 0.25,
                 angular_threshold: float = 0.1,
                 α: float = 1.0,    # forward‑progress weight
                 β: float = 5.0,    # cross‑track penalty weight
                 γ: float = 2.0     # heading‑alignment weight
                 ):
        self.threshold = threshold
        self.angular_threshold = angular_threshold
        self.α = α
        self.β = β
        self.γ = γ
        # store last potential
        self.last_phi = None
        # store last segment id so we can reset shaping when waypoint changes
        self._last_waypoint_hash = None

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def compute_potential(self, offset_curr, offset_last, yaw):
        """
        offset_curr = [offset_x, offset_y]: vector from AUV to current waypoint
        offset_last = [offset_x_last, offset_y_last]: vector from AUV to previous waypoint
        yaw = current heading theta
        """
        # Segment vector from prev → curr:
        seg = np.array(offset_curr) - np.array(offset_last)
        norm = np.linalg.norm(seg) + 1e-8
        seg_u = seg / norm

        # Position of AUV relative to prev waypoint:
        #   prev_wp = pos + offset_last  →  pos_rel_prev = pos - prev_wp = -offset_last
        pos_rel_prev = -np.array(offset_last)

        # forward‐along‐track coordinate
        fwd = pos_rel_prev.dot(seg_u)

        # cross‐track error
        perp = pos_rel_prev - fwd*seg_u
        e_ct = np.linalg.norm(perp)

        # segment heading
        seg_theta = np.arctan2(seg_u[1], seg_u[0])

        # heading error
        h_err = self.wrap_to_pi(yaw - seg_theta)

        # potential
        return self.α * fwd - self.β * e_ct + self.γ * np.cos(h_err)

    def get_reward(
        self,
        distance_to_goal, theta, target_theta,
        action_magnitude, number_of_steps,
        dot_to_goal=0.0, last_distance_to_goal=0.0,
        offset_x=0.0, offset_y=0.0, offset_z=0.0,
        offset_x_last=0.0, offset_y_last=0.0, offset_z_last=0.0,
        last_closest_distance_to_goal=0.0,
        terminated=False,
    ):
        # 1) completion / termination
        r_complete = 0.0
        if distance_to_goal < self.threshold:
            r_complete += 1500.0
        if terminated:
            r_complete -= 500.0

        # 2) compute a hash of (offset_curr, offset_last) to detect new segment
        #    this is just to reset shaping on waypoint change
        waypoint_hash = (round(offset_x,3), round(offset_y,3),
                         round(offset_x_last,3), round(offset_y_last,3))
        new_segment = (waypoint_hash != self._last_waypoint_hash)
        self._last_waypoint_hash = waypoint_hash

        # 3) compute current potential
        offset_curr = [offset_x, offset_y]
        offset_last = [offset_x_last, offset_y_last]
        phi = self.compute_potential(offset_curr, offset_last, theta)

        # on a new segment, reset last_phi so first Δφ=0
        if new_segment or self.last_phi is None:
            self.last_phi = phi

        # 4) shaping = Δφ
        r_shaping = phi - self.last_phi
        self.last_phi = phi

        # 5) combine with a small time‐step penalty and action penalty on translation only
        time_pen = -0.05 * number_of_steps

        # action_magnitude might be scalar or vector
        act = action_magnitude
        try:
            if len(act) >= 3:
                t_mag = np.linalg.norm(np.array(act[:3]))
            else:
                t_mag = np.linalg.norm(np.array(act))
        except TypeError:
            t_mag = float(act)
        action_pen = np.exp(-(t_mag**2))

        total = r_complete + r_shaping + time_pen + action_pen

        return total, np.array([r_complete, r_shaping, time_pen, action_pen])

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
