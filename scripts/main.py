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

        for row in reader:
            if len(row) >= 3:
                try:
                    x, y, z, theta = (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                    )
                    waypoints.append([x, y, z, theta])
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
    env = gym.make(
        "BlueRov-v0",
        render_mode="human",
        trajectory_file=trajectory_file,
        max_episode_steps=max_steps,
    )

    base_env = env.unwrapped

    exit()

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

    dt = 0.1
    proximity_threshold = 0.1
    current_waypoint_idx = 1
    step_count = 0

    print(f"\nStarting trajectory tracking with {len(waypoints)} waypoints")
    print(
        f"Initial position: x={base_env.state['x']:.2f}, y={base_env.state['y']:.2f}, z={base_env.state['z']:.2f}"
    )
    print(f"Initial heading: {base_env.state['theta']:.2f} rad")
    print(f"\nWaiting for 5 seconds to stabilize the simulation...")
    time.sleep(5)

    while step_count < max_steps:
        # Use true state for control
        current_pos = np.array(
            [base_env.state["x"], base_env.state["y"], base_env.state["z"]]
        )
        current_heading = base_env.state["theta"]

        target_waypoint = waypoints[current_waypoint_idx]
        target_pos = np.array(target_waypoint[:3])
        target_theta = target_waypoint[3]

        # Calculate position errors
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        error_z = target_pos[2] - current_pos[2]

        # Calculate heading error (accounting for angle wrapping)
        error_heading = np.arctan2(
            np.sin(target_theta - current_heading),
            np.cos(target_theta - current_heading),
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

        # Print status
        distance_to_target = np.linalg.norm(current_pos - target_pos)
        heading_error_to_target = np.arctan2(
            np.sin(target_theta - current_heading),
            np.cos(target_theta - current_heading),
        )

        print(
            f"Step {step_count}: Position (x={base_env.state['x']:.2f}, y={base_env.state['y']:.2f}, z={base_env.state['z']:.2f})"
        )
        print(
            f"Heading: {base_env.state['theta']:.2f} rad, Target: {target_theta:.2f} rad, Error: {heading_error_to_target:.2f} rad"
        )
        print(
            f"Current waypoint: {current_waypoint_idx}, Distance: {distance_to_target:.2f}"
        )
        print(
            f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]"
        )

        # Check if we've reached the current waypoint
        if (
            distance_to_target < proximity_threshold
            and abs(heading_error_to_target) < 0.1
        ):
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
            f"Offset: x={obs['offset_x'][0]:.2f}, y={obs['offset_y'][0]:.2f}, z={obs['offset_z'][0]:.2f}"
        )
        print(f"Reward: {reward:.2f}")

        if terminated or truncated:
            obs, _ = env.reset()
            print("Episode ended, resetting...")

        step_count += 1

    env.close()


def main():
    parser = argparse.ArgumentParser(description="BlueROV2 Control and Simulation")

    parser.add_argument("--file", type=str, help="Path to trajectory CSV file")

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["pid", "manual"],
        default="pid",
        help="Control algorithm to use (pid, ppo, sac, a2c, or manual)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum number of steps per episode (default: 100000)",
    )

    args = parser.parse_args()

    # Run the appropriate controller
    if args.algorithm == "pid":
        if not args.file:
            print("Error: PID controller requires a trajectory file (--file)")
            return
        run_pid_controller(args.file, args.max_steps)
    elif args.algorithm == "manual":
        manual_control(args.max_steps)


if __name__ == "__main__":
    main()
