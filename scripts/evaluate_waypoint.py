import time
from pathlib import Path

import typer
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bluerov2_gym


def evaluate(
    model_path: str, trajectory_path: str, num_episodes: int = 10, render: bool = True
):
    """
    Evaluate a trained model on trajectory following

    Args:
        model_path: Path to trained model
        trajectory_path: Path to CSV file with trajectory waypoints
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    print(f"Evaluating model: {model_path}")
    print(f"On trajectory: {trajectory_path}")

    # Load the model
    model = PPO.load(model_path)

    # Create the environment with waypoint rewards and rendering if needed
    render_mode = "human" if render else None
    env = gym.make(
        "BlueRov-v0",
        trajectory_path=trajectory_path,
        waypoint_reward=True,
        render_mode=render_mode,
    )

    # Load the normalization stats
    # We use DummyVecEnv with a single environment for evaluation
    # regardless of how many envs were used during training
    vec_env = DummyVecEnv(
        [
            lambda: gym.make(
                "BlueRov-v0", trajectory_path=trajectory_path, waypoint_reward=True
            )
        ]
    )
    vec_env = VecNormalize.load(str(model_path) + "_vec_normalize.pkl", vec_env)
    vec_env.training = False  # No training, only inference
    vec_env.norm_reward = False

    total_waypoints_reached = 0
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        if render:
            env.render()

        done = False
        episode_reward = 0
        step_count = 0

        print(f"\nStarting evaluation episode {episode + 1}")

        while not done:
            # Normalize the observation using the loaded statistics
            obs_dict = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
            obs_normalized = vec_env.normalize_obs(obs_dict)

            # Get the action from the trained model
            action, _ = model.predict(obs_normalized, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render:
                env.unwrapped.step_sim()
                time.sleep(0.01)  # Small delay for visualization

            step_count += 1

            # Periodically print progress
            if step_count % 10 == 0:
                waypoint_progress = info.get("waypoint_progress", 0.0)
                print(
                    f"  Step {step_count}: Waypoint progress: {waypoint_progress*100:.1f}%"
                )

        # Print episode summary
        waypoints_reached = (
            info.get("waypoint_progress", 0.0) * env.unwrapped.reward_fn.total_waypoints
        )
        total_waypoints_reached += waypoints_reached
        total_rewards.append(episode_reward)

        print(f"Episode {episode + 1} finished after {step_count} steps")
        print(f"Total reward: {episode_reward:.2f}")
        print(
            f"Waypoints reached: {waypoints_reached:.1f}/{env.unwrapped.reward_fn.total_waypoints}"
        )

    # Print overall evaluation results
    print("\n===== Evaluation Results =====")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(
        f"Average waypoints reached: {total_waypoints_reached/num_episodes:.1f}/{env.unwrapped.reward_fn.total_waypoints}"
    )
    print(
        f"Success rate (completed trajectory): {sum([r > 0 for r in total_rewards])/num_episodes*100:.1f}%"
    )

    env.close()
    vec_env.close()

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "avg_waypoints_reached": float(total_waypoints_reached / num_episodes),
        "success_rate": float(sum([r > 0 for r in total_rewards]) / num_episodes * 100),
    }


# Create Typer app
app = typer.Typer(help="Evaluate BlueRov2 models on trajectory following")


@app.command("evaluate")
def evaluate_command(
    model: str = typer.Argument(..., help="Path to trained model"),
    trajectory: str = typer.Argument(..., help="Path to CSV trajectory file"),
    num_episodes: int = typer.Option(10, help="Number of evaluation episodes"),
    no_render: bool = typer.Option(False, help="Disable rendering during evaluation"),
):
    """Evaluate a trained model on trajectory following"""
    evaluate(
        model_path=model,
        trajectory_path=trajectory,
        num_episodes=num_episodes,
        render=not no_render,
    )


if __name__ == "__main__":
    app()
