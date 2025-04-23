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
        
    # You can add additional reward functions here for different tasks
    # For example: trajectory following, station keeping, etc.
