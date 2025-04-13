from stable_baselines3.common.callbacks import BaseCallback


class WaypointTrainingCallback(BaseCallback):
    """
    Custom callback for printing waypoint training progress
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_reward = 0
        self.last_waypoint_progress = 0

    def _on_step(self):
        # Extract info from the environment
        info = self.locals.get("infos")[0]
        reward = self.locals.get("rewards")[0]

        self.episode_reward += reward

        # Check if episode ended and log results
        done = self.locals.get("dones")[0]
        if done:
            self.episode_count += 1
            waypoint_progress = info.get("waypoint_progress", 0.0)

            if self.verbose > 0:
                print(f"----- Episode {self.episode_count} completed -----")
                print(f"Total reward: {self.episode_reward:.2f}")
                print(
                    f"Waypoint progress: {waypoint_progress*100:.1f}% of trajectory completed"
                )
                print("--------------------------------------")

            self.episode_reward = 0
            self.last_waypoint_progress = waypoint_progress

        return True
