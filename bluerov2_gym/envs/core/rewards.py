import numpy as np


class Reward:
    """
    Reward Function for BlueROV2 Reinforcement Learning Environment
    
    This class defines the reward function used to evaluate the performance
    of the agent in the BlueROV2 environment. The reward is designed to
    encourage the agent to minimize position errors, maintain stable velocities,
    and keep the desired orientation.
    """
    
    def __init__(self):
        """
        Initialize the reward function.
        You can add custom reward parameters or configurations here.
        """
        # Currently no parameters needed in initialization

        self.dt          = 0.1                                   # [s]
        self.k_cross     =  2.0    # distance weight  (-)
        self.k_align     =  0.5    # velocity-alignment weight
        self.k_head      =  0.2    # heading-error weight
        self.k_ctrl      =  0.01   # control-effort weight
        self.lookahead   =  5      # how many steps ahead to define desired direction
        self.last_s      =  0.0    # arc-length travelled at previous step
        pass

    def get_reward(self, obs):
        """
        Calculate the reward based on the current observation.
        
        The reward is a negative value that penalizes:
        1. Distance from origin (position error)
        2. High velocities (velocity penalty)
        3. Deviation from desired orientation (orientation error)
        
        Args:
            obs (dict): Observation dictionary containing state variables
            
        Returns:
            float: Calculated reward value (negative value where higher/closer to zero is better)
        """
        # Calculate Euclidean distance from origin (position error)
        position_error = np.sqrt(obs["x"][0] ** 2 + obs["y"][0] ** 2 + obs["z"][0] ** 2)

        # Calculate velocity magnitude as a penalty for high-speed movements
        velocity_penalty = np.sqrt(
            obs["vx"][0] ** 2 + obs["vy"][0] ** 2 + obs["vz"][0] ** 2
        )

        # Calculate orientation error (deviation from desired heading)
        orientation_error = abs(obs["theta"][0])

        # Combined reward with different weights for each component
        reward = -(
            1.0 * position_error    # Weight for position error
            + 0.1 * velocity_penalty  # Weight for velocity (smaller weight)
            + 0.5 * orientation_error  # Weight for orientation
        )

        return reward

    def get_reward_trajectory(self, obs, action, trajectory_file):
        """
        Dense reward that guides the agent along a pre-defined 3-D trajectory.

        obs must contain:
            x,y,z            position [m]
            vx,vy,vz         body-frame or world-frame velocity [m/s]
            yaw              heading angle (rad)  **only needed for heading term**

        action: vector of motor/thruster commands (used only for effort penalty)
        """
        self.reference = trajectory_file    
        self.N_ref       = len(self.reference)
        # ------------------------------------------------------------------
        # 2-A.  Find the *nearest* point on the trajectory and its index i*
        # ------------------------------------------------------------------
        pos = np.array([obs["x"][0],  obs["y"][0],  obs["z"][0]])
        dists  = np.linalg.norm(self.reference - pos, axis=1)
        i_star = int(np.argmin(dists))                               # closest waypoint
        p_star = self.reference[i_star]

        # cross-track (perpendicular) error
        cross_track = np.linalg.norm(pos - p_star)                   # e⊥
        r_cross = -self.k_cross * cross_track                        # (negative)

        # ------------------------------------------------------------------
        # 2-B.  Desired *forward* direction (lookahead few steps so it’s smooth)
        # ------------------------------------------------------------------
        i_forward   = min(i_star + self.lookahead, self.N_ref - 1)
        p_forward   = self.reference[i_forward]
        desired_dir = p_forward - p_star
        desired_dir /= np.linalg.norm(desired_dir) + 1e-8            # unit vector

        #  Align current velocity with the desired direction
        vel         = np.array([obs["vx"][0], obs["vy"][0], obs["vz"][0]])
        vel_mag     = np.linalg.norm(vel) + 1e-8
        r_align     =  self.k_align * np.dot(vel, desired_dir) / vel_mag

        #  Optional: heading penalty if you have yaw in your state
        head_err    = Reward.angle_wrap(np.arctan2(desired_dir[1], desired_dir[0]) -
                                obs["theta"][0])
        r_heading   = -self.k_head * abs(head_err)

        # ------------------------------------------------------------------
        # 2-C.  Small penalty on control effort
        # ------------------------------------------------------------------
        r_ctrl      = -self.k_ctrl * np.square(action).sum()

        # ------------------------------------------------------------------
        # 2-D.  Total reward
        # ------------------------------------------------------------------
        reward = r_cross + r_align + r_heading + r_ctrl
        # print(reward)
        return reward
        
    @staticmethod
    def angle_wrap(a):
        """
        Wrap an angle to the range [-pi, pi].
        
        Args:
            angle (float): Angle in radians
            
        Returns:
            float: Wrapped angle in radians
        """
        return (a + np.pi) % (2 * np.pi) - np.pi