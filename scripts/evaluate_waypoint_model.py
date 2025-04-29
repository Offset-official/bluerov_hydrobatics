import time
import argparse
import gymnasium as gym
import bluerov2_gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize


def evaluate(
    model_path: str,
    num_episodes: int,
    model_type: str,
    normalization_file: str = None,
    trajectory_file: str = None,
):
    env0 = DummyVecEnv(
        [
            lambda: gym.make(
                "BlueRov-v0", render_mode="human", trajectory_file=trajectory_file
            )
        ]
    )
    env = VecNormalize.load(normalization_file, env0)
    env.training = False

    if model_type.lower() == "ppo":
        model = PPO.load(model_path)
    elif model_type.lower() == "sac":
        model = SAC.load(model_path)
    elif model_type.lower() == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    obs = env.reset()
    done = False
    total_reward = 0.0
    success = False
    step_count = 0
    env.render()
    time.sleep(5)

    num_waypoints = None
    if trajectory_file is not None:
        import numpy as np

        num_waypoints = np.loadtxt(trajectory_file, delimiter=",").shape[0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        total_reward += sum(info[0]["reward_tuple"])
        env.unwrapped.env_method("step_sim")
        step_count += 1
        if trajectory_file is not None:
            base_env = env.venv
            waypoint_idx = getattr(base_env, "waypoint_idx", None)
            if waypoint_idx is not None and waypoint_idx >= num_waypoints:
                success = True
                print(f"All waypoints completed in {step_count} steps.")
                break

    print("\n=== Evaluation Summary ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success: {success}")
    print(f"Steps taken: {step_count}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO or SAC model on BlueRov-v0 and report metrics"
    )
    parser.add_argument(
        "--model-path",
        help="Path to the saved model (e.g. ./trained_models/bluerov_simplepoint.zip)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation",
    )
    parser.add_argument(
        "--normalization-file",
        type=str,
        help="Path to the normalization file (optional)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "a2c"],
        help="Type of RL model to evaluate (ppo or sac)",
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="Path to the trajectory file (optional)",
    )
    args = parser.parse_args()

    evaluate(
        args.model_path,
        args.num_episodes,
        args.model_type,
        args.normalization_file,
        args.trajectory_file,
    )
