import time
import csv
import argparse
import numpy as np
import gymnasium as gym
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import matplotlib.pyplot as plt

import bluerov2_gym  # This import will automatically register the environment

# Add imports for Stable Baselines
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Initialize error terms
        self.error_sum = 0
        self.prev_error = 0

    def compute(self, error, dt):
        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_sum += error * dt
        i_term = self.ki * self.error_sum

        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        # Total control output
        control = p_term + i_term + d_term

        return control

    def reset(self):
        """Reset the error history"""
        self.error_sum = 0
        self.prev_error = 0


def load_trajectory_from_csv(file_path):
    waypoints = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        # Verify the CSV format
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

    return np.array(waypoints)


def calculate_heading(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Default heading if points are too close
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0

    return np.arctan2(dy, dx)



def run_pid_controller(trajectory_file, max_steps=100000):
    # Define trajectory
    waypoints = load_trajectory_from_csv(trajectory_file)
    if len(waypoints) == 0:
        print("No valid waypoints found in trajectory file.")
        return

    # Create the environment with rendering enabled and increased time limit
    env = gym.make("BlueRov-v0", render_mode="human",trajectory_file = trajectory_file, max_episode_steps=max_steps)

    obs = env.reset()[0]
    env.render()

    # Initialize PID controllers with tuned parameters
    pid_x = PIDController(kp=1.0, ki=0.0, kd=0.0)
    pid_y = PIDController(kp=1.0, ki=0.0, kd=0.0)
    pid_z = PIDController(kp=1.0, ki=0.0, kd=0.0)
    pid_heading = PIDController(kp=1.0, ki=0.0, kd=0.0)

    # Reset all controllers
    pid_x.reset()
    pid_y.reset()
    pid_z.reset()
    pid_heading.reset()

    # Time step
    dt = 0.1

    # Proximity threshold to consider a waypoint reached
    proximity_threshold = 0.1

    # Current waypoint index (start at 1 since we're already at waypoint 0)
    current_waypoint_idx = 1

    # Main control loop
    step_count = 0

    print(f"\nStarting trajectory tracking with {len(waypoints)} waypoints")
    print(
        f"Initial position: x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f}"
    )
    print(f"Initial heading: {obs['theta'][0]:.2f} rad")
    print(f"\nWaiting for 5 seconds to stabilize the simulation...")
    time.sleep(5)

    while step_count < max_steps:
        current_pos = np.array([obs["x"][0], obs["y"][0], obs["z"][0]])
        current_heading = obs["theta"][0]

        target_waypoint = waypoints[current_waypoint_idx]

        # For smoother motion, use the path direction rather than direct heading to target
        if current_waypoint_idx < len(waypoints) - 1:
            # Calculate heading from current waypoint to next waypoint
            path_heading = calculate_heading(
                waypoints[current_waypoint_idx], waypoints[current_waypoint_idx + 1]
            )
        else:
            # For the last waypoint, just head directly to it
            path_heading = calculate_heading(current_pos, target_waypoint)

        # Calculate position errors
        error_x = target_waypoint[0] - current_pos[0]
        error_y = target_waypoint[1] - current_pos[1]
        error_z = target_waypoint[2] - current_pos[2]

        # Calculate heading error (accounting for angle wrapping)
        error_heading = np.arctan2(
            np.sin(path_heading - current_heading),
            np.cos(path_heading - current_heading),
        )

        # Compute PID control outputs
        control_x = pid_x.compute(error_x, dt)
        control_y = pid_y.compute(error_y, dt)
        control_z = pid_z.compute(error_z, dt)
        control_heading = pid_heading.compute(error_heading, dt)

        # Convert to BlueROV action space
        # [forward, lateral, vertical, rotation]
        # Transform x, y controls to forward/lateral based on current heading
        forward = control_x * np.cos(current_heading) + control_y * np.sin(
            current_heading
        )
        lateral = -control_x * np.sin(current_heading) + control_y * np.cos(
            current_heading
        )

        action = np.clip(
            [
                forward,  # Forward/backward
                lateral,  # Left/right
                control_z,  # Up/down
                control_heading,  # Rotation
            ],
            -1.0,
            1.0,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.2f}")
        env.unwrapped.step_sim()

        # Print status
        distance_to_target = np.linalg.norm(current_pos - target_waypoint)
        print(
            f"Step {step_count}: Position (x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f})"
        )
        print(
            f"Heading: {obs['theta'][0]:.2f} rad, Desired: {path_heading:.2f} rad, Error: {error_heading:.2f} rad"
        )
        print(
            f"Current waypoint: {current_waypoint_idx}, Distance: {distance_to_target:.2f}"
        )
        print(
            f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]"
        )

        # Check if we've reached the current waypoint
        if distance_to_target < proximity_threshold:
            print(f"Reached waypoint {current_waypoint_idx}")
            current_waypoint_idx += 1

            pid_x.reset()
            pid_y.reset()
            pid_z.reset()
            pid_heading.reset()

            if current_waypoint_idx >= len(waypoints):
                print("Trajectory completed!")
                break

        # Check if episode ended
        if terminated or truncated:
            print("Episode terminated or truncated")
            break

        time.sleep(0.1)

        step_count += 1

    env.close()


def run_rl_agent(algorithm, model_path=None, trajectory_file=None, max_steps=100000):
    """Run BlueROV with a specified RL algorithm"""
    # Create the environment with rendering enabled and increased step limit
    env = gym.make("BlueRov-v0", render_mode="human", max_episode_steps=max_steps)

    # If trajectory file is provided, visualize it
    if trajectory_file:
        waypoints = load_trajectory_from_csv(trajectory_file)
        if len(waypoints) > 0:
            obs, _ = env.reset()
            vis = env.unwrapped.renderer.vis
            visualize_trajectory(vis, waypoints)

            # Optionally, start from first waypoint
            obs = set_initial_state(env, waypoints)
        else:
            obs, _ = env.reset()
    else:
        obs, _ = env.reset()

    # Render initial state
    env.render()

    algorithm = algorithm.lower()

    # Check if we're loading a pre-trained model
    if model_path:
        print(f"Loading pre-trained {algorithm.upper()} model from {model_path}")

        # Load the appropriate model based on algorithm
        if algorithm == "ppo":
            model = PPO.load(model_path)
        else:
            print(f"Unknown algorithm: {algorithm}. Using PPO as default.")
            model = PPO.load(model_path)

        # Try to load normalization stats if available
        try:
            normalize_path = f"{model_path}_vec_normalize.pkl"
            vec_env = DummyVecEnv(
                [lambda: gym.make("BlueRov-v0", max_episode_steps=max_steps)]
            )
            vec_env = VecNormalize.load(normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            use_normalization = True
            print("Successfully loaded normalization stats")
        except FileNotFoundError:
            use_normalization = False
            print("No normalization stats found, running without normalization")
    else:
        print(f"No model path provided for {algorithm}. Running with random actions.")
        model = None
        use_normalization = False

    # Run episodes
    episodes = 1  # Default to one episode

    for episode in range(episodes):
        episode_reward = 0
        step_count = 0

        print(f"\nStarting Episode {episode + 1}")

        while step_count < max_steps:
            if model:
                if use_normalization:
                    if isinstance(obs, dict):
                        obs_array = np.concatenate(
                            [obs[key] for key in sorted(obs.keys())]
                        )
                    else:
                        obs_array = obs

                    obs_normalized = vec_env.normalize_obs(obs_array)
                    action, _ = model.predict(obs_normalized, deterministic=True)
                else:
                    if isinstance(obs, dict):
                        obs_array = np.concatenate(
                            [obs[key] for key in sorted(obs.keys())]
                        )
                    else:
                        obs_array = obs

                    action, _ = model.predict(obs_array, deterministic=True)
            else:
                # Random action if no model is loaded
                action = np.random.uniform(-1, 1, 4)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            env.unwrapped.step_sim()

            # Add a small delay to make the visualization viewable
            time.sleep(0.1)

            step_count += 1

            # Print current state
            print(
                f"Step {step_count}: Position (x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f})"
            )
            print(f"Current reward: {reward:.2f}")
            print(f"Action: {action}")

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step_count} steps")
                print(f"Total reward: {episode_reward:.2f}")
                break

    env.close()


def manual_control(max_steps=100000):
    """
    Test the environment with manual controls for debugging
    Keys:
    - W/S: Forward/Backward
    - A/D: Left/Right
    - Q/E: Rotate
    - R/F: Up/Down
    """
    env = gym.make("BlueRov-v0", render_mode="human", max_episode_steps=max_steps)
    obs, _ = env.reset()
    env.render()

    step_count = 0

    print("\nManual Control Mode")
    print("Controls:")
    print("- W/S: Forward/Backward")
    print("- A/D: Left/Right")
    print("- Q/E: Rotate")
    print("- R/F: Up/Down")
    print("- X: Exit")

    while step_count < max_steps:
        action = np.array([0.0, 0.0, 0.0, 0.0])

        key = input("Enter control (wasdqerf, x to exit): ").lower()

        if key == "x":
            break
        elif key == "w":
            action[0] = 1.0  # Forward
        elif key == "s":
            action[0] = -1.0  # Backward
        elif key == "a":
            action[1] = -1.0  # Left
        elif key == "d":
            action[1] = 1.0  # Right
        elif key == "q":
            action[3] = -1.0  # Rotate left
        elif key == "e":
            action[3] = 1.0  # Rotate right
        elif key == "r":
            action[2] = 1.0  # Up
        elif key == "f":
            action[2] = -1.0  # Down

        print(f"Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)

        env.unwrapped.step_sim()

        print(
            f"Position: x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f}"
        )
        print(f"Reward: {reward:.2f}")

        if terminated or truncated:
            obs, _ = env.reset()
            print("Episode ended, resetting...")

        step_count += 1

    env.close()


def plot_trajectory(trajectory):
    """
    Plot the inputted trajectory in 3D space.

    Args:
        trajectory (numpy.ndarray): Array of shape (num_points, 3) containing x, y, z coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Plot the trajectory
    ax.plot(x, y, z, label='Trajectory', color='blue')

    # Highlight start and end points
    ax.scatter(x[0], y[0], z[0], color='green', label='Start', s=100)
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=100)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Inputted Trajectory')
    ax.legend()

    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="BlueROV2 Control and Simulation")

    parser.add_argument("--file", type=str, help="Path to trajectory CSV file")

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["pid", "ppo", "manual"],
        default="pid",
        help="Control algorithm to use (pid, ppo, sac, td3, a2c, or manual)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to the pre-trained model (required for RL algorithms)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum number of steps per episode (default: 100000)",
    )

    args = parser.parse_args()

    # Validate arguments
    if (
        args.algorithm in ["ppo"]
        and not args.model
        and args.algorithm != "manual"
    ):
        print(
            f"Warning: No model provided for {args.algorithm}. Will run with random actions."
        )

    # Run the appropriate controller
    if args.algorithm == "pid":
        if not args.file:
            print("Error: PID controller requires a trajectory file (--file)")
            return
        run_pid_controller(args.file, args.max_steps)
    elif args.algorithm == "manual":
        manual_control(args.max_steps)
    else:  # RL algorithms
        run_rl_agent(args.algorithm, args.model, args.file, args.max_steps)


if __name__ == "__main__":
    main()
