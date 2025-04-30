import time
import csv
import argparse
import numpy as np
import gymnasium as gym
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import matplotlib.pyplot as plt
<<<<<<< HEAD

from pathlib import Path
import bluerov2_gym  # This import will automatically register the environment
from gymnasium import spaces   # add with other imports
# Add imports for Stable Baselines
from stable_baselines3 import PPO, SAC, TD3, A2C
=======
import bluerov2_gym  # This import will automatically register the environment
>>>>>>> b190b04c2e6505bd4ed6aa016bf6d85d0bf84167
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def maybe_flatten(obs):
    """Return obs unchanged if it's already a Box; flatten Dict -> array."""
    if isinstance(obs, dict):
        return np.concatenate([obs[k] for k in sorted(obs.keys())])
    return obs
ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "a2c": A2C,
}

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

class TrajectoryReward(gym.Wrapper):
    def __init__(self, env, waypoints, proximity=0.3):
        super().__init__(env)
        self.waypoints = np.asarray(waypoints)
        self.idx = 0
        self.proximity = proximity

<<<<<<< HEAD
    def reset(self, **kwargs):
        self.idx = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        pos = np.array([obs["x"][0], obs["y"][0], obs["z"][0]])
        target = self.waypoints[self.idx]

        # distance‑based reward (closer is better)
        dist = np.linalg.norm(pos - target)
        reward = -dist

        # small bonus for heading roughly along the path
        if self.idx < len(self.waypoints) - 1:
            desired_vec = self.waypoints[self.idx + 1] - target
            heading_vec = np.array([np.cos(obs["theta"][0]), np.sin(obs["theta"][0]), 0])
            reward += 0.1 * np.dot(desired_vec[:2], heading_vec[:2]) / (np.linalg.norm(desired_vec[:2]) + 1e-6)

        # progress to next waypoint
        if dist < self.proximity:
            reward += 10                      # waypoint bonus
            self.idx += 1
            if self.idx == len(self.waypoints):
                terminated = True             # finished!

        return obs, reward, terminated, truncated, info

def train_rl_agent(
    algorithm: str,
    total_timesteps: int,
    save_model_path: str,
    normalize: bool = False,
    trajectory_file: str | None = None,
    max_steps: int = 100_000,
    eval_freq: int = 25_000,
) -> None:
    """
    Train an RL policy on **BlueRov-v0** with Stable‑Baselines3.

    Parameters
    ----------
    algorithm : str
        One of {"ppo", "sac", "td3", "a2c"} (case‑insensitive).
    total_timesteps : int
        Number of environment timesteps to train for.
    save_model_path : str
        Directory prefix where the model.zip *and* (optionally) vec_normalize.pkl
        will be written.  If the directory does not exist, it is created.
    normalize : bool, default=False
        Wrap the env with VecNormalize (recommended for continuous control).
    trajectory_file : str | None
        (Optional) CSV of way‑points – for training.
    max_steps : int
        Per‑episode time‑limit passed to the env.
    eval_freq : int
        Evaluate/Checkpoint every `eval_freq` timesteps.
    ---------------------------------------------------------------------------
    How to run
    ----------
    python main.py --train --algorithm ppo --total-timesteps 2_000_000 \\
                   --save-model models/ppo_bluerov --normalize

    # Resume (will load model.zip + (optional) vec_normalize.pkl automatically)
    python main.py --train --algorithm ppo --total-timesteps 1_000_000 \\
                   --save-model models/ppo_bluerov --normalize
    """

    algorithm = algorithm.lower()
    if algorithm not in ALGO_MAP:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    save_dir = Path(save_model_path).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Env creation ------------------------------------------------
    def _make_env():
        e = gym.make("BlueRov-v0", max_episode_steps=max_steps)
        if trajectory_file:
            wpts = load_trajectory_from_csv(trajectory_file)
            e = TrajectoryReward(e, wpts)
        return e

    if normalize:
        env = DummyVecEnv([_make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    else:
        env = DummyVecEnv([_make_env])

    if isinstance(env.observation_space, spaces.Dict):
        policy_name = "MultiInputPolicy"      # ←  handles Dict observations
    else:
        policy_name = "MlpPolicy"

    # ---------------- Model (create‑or‑resume) ------------------------------------
    model_path = save_dir / "model.zip"
    vecnorm_path = save_dir / "model_vec_normalize.pkl"

    if model_path.exists():
        print(f"Resuming training from {model_path}")
        
        model = ALGO_MAP[algorithm](
            policy_name,
            env,
            verbose=1,
            tensorboard_log=str(save_dir / "tb"),
        )
        # If we were normalising, reload stats
        if normalize and vecnorm_path.exists():
            print("Loading VecNormalize statistics")
            env = VecNormalize.load(str(vecnorm_path), env)
            env.training = True
            env.norm_reward = True
            model.set_env(env)
    else:
        print(f"Starting fresh {algorithm.upper()} training run")
        
        model = ALGO_MAP[algorithm](
            policy_name,
            env,
            verbose=1,
            tensorboard_log=str(save_dir / "tb"),
        )

    # ---------------- Callbacks (eval + checkpoints) ------------------------------
    eval_env = DummyVecEnv([_make_env])
    if normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=normalize,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    # ---------------- Train --------------------------------------------------------
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])


    # ---------------- Save artefacts ----------------------------------------------
    print(f"Training finished – saving to {model_path}")
    model.save(str(model_path))
    if normalize:
        env.save(str(vecnorm_path))
=======
def main():
    parser = argparse.ArgumentParser(description="BlueROV2 Control and Simulation")
>>>>>>> b190b04c2e6505bd4ed6aa016bf6d85d0bf84167

    env.close()
    eval_env.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="BlueROV2 Control, Evaluation and Training Harness"
    )

    # Top‑level *mode*
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode (default: run/evaluate).",
    )

    # Common options
    parser.add_argument(
        "--algorithm",
<<<<<<< HEAD
        type=str.lower,
        choices=list(ALGO_MAP.keys()) + ["pid", "manual"],
        default="pid",
        help="Control algorithm.",
    )
    parser.add_argument("--file", type=str, help="Path to trajectory CSV.")
=======
        type=str,
        choices=["pid", "manual"],
        default="pid",
        help="Control algorithm to use (pid, ppo, sac, a2c, or manual)",
    )

>>>>>>> b190b04c2e6505bd4ed6aa016bf6d85d0bf84167
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100_000,
        help="Maximum steps per episode (env time‑limit).",
    )

    # ---------- (run/evaluate) specific -------------------------------------
    parser.add_argument("--model", type=str, help="Path prefix of trained model.")

<<<<<<< HEAD
    # ---------- (train) specific --------------------------------------------
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total timesteps for training (only used with --train).",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="models/default_bluerov_model",
        help="Where to save checkpoints/best/model.zip (only used with --train).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Wrap env in VecNormalize during training.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="How often (timesteps) to checkpoint/evaluate when training.",
    )

    return parser.parse_args()



def main():
    args = parse_args()

    # ------------------------------ PID / Manual ------------------------------
    if args.algorithm in {"pid"} and not args.train:
=======
    # Run the appropriate controller
    if args.algorithm == "pid":
>>>>>>> b190b04c2e6505bd4ed6aa016bf6d85d0bf84167
        if not args.file:
            print("Error: --algorithm pid requires --file path/to/trajectory.csv")
            return
        run_pid_controller(args.file, args.max_steps)
        return
    if args.algorithm == "manual" and not args.train:
        manual_control(args.max_steps)
<<<<<<< HEAD
        return

    # ------------------------------ TRAIN -------------------------------------
    if args.train:
        train_rl_agent(
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps,
            save_model_path=args.save_model,
            normalize=args.normalize,
            trajectory_file=args.file,
            max_steps=args.max_steps,
            eval_freq=args.eval_freq,
        )
        return

    # ------------------------------ RUN / EVAL -------------------------------
    run_rl_agent(
        algorithm=args.algorithm,
        model_path=args.model,
        trajectory_file=args.file,
        max_steps=args.max_steps,
    )
=======
>>>>>>> b190b04c2e6505bd4ed6aa016bf6d85d0bf84167


if __name__ == "__main__":
    main()