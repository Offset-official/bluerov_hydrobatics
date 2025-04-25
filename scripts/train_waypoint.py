import time
from pathlib import Path

import typer
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.utils import set_random_seed

from bluerov2_gym.training.callbacks import WaypointTrainingCallback


def make_env(trajectory_path, render_mode, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param trajectory_path: Path to the trajectory CSV file
    :param waypoint_threshold: Distance threshold for considering a waypoint reached
    :param rank: Index of the subprocess
    :param seed: The initial seed for RNG
    """

    def _init():
        env = gym.make(
            "BlueRov-v0",
            trajectory_file=trajectory_path,
            render_mode=render_mode,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def train(
    trajectory_path: str,
    output_dir: str = "./trained_models",
    total_timesteps: int = 1000000,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    waypoint_threshold: float = 0.01,
    n_envs: int = 8,
    render_mode: str = "none",
):
    """
    Train an agent to follow a trajectory using PPO with parallel environments

    Args:
        trajectory_path: Path to CSV file with trajectory waypoints
        output_dir: Directory to save trained models
        total_timesteps: Total timesteps for training
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        waypoint_threshold: Distance threshold for considering a waypoint reached
        n_envs: Number of parallel environments
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_name = Path(trajectory_path).stem

    model_name = f"bluerov_waypoint_{trajectory_name}"
    model_path = output_dir / model_name

    print(f"Training on trajectory: {trajectory_path}")
    print(f"Model will be saved to: {model_path}")
    print(f"Using {n_envs} parallel environments")

    # Create vectorized environment with multiple parallel envs
    env = SubprocVecEnv(
        [make_env(trajectory_path, render_mode, i) for i in range(n_envs)]
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=model_name,
    )

    waypoint_callback = WaypointTrainingCallback()

    print("Starting training...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, waypoint_callback],
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    model.save(model_path)
    env.save(str(model_path) + "_vec_normalize.pkl")
    print(f"Model saved to {model_path}")

    return model


app = typer.Typer(help="Train BlueRov2 on trajectory following")


@app.command("train")
def train_command(
    trajectory: str = typer.Argument(..., help="Path to CSV trajectory file"),
    output_dir: str = typer.Option(
        "./trained_models", help="Directory to save trained models"
    ),
    timesteps: int = typer.Option(1000000, help="Total timesteps for training"),
    threshold: float = typer.Option(
        0.5, help="Distance threshold for considering a waypoint reached"
    ),
    learning_rate: float = typer.Option(0.01, help="Learning rate for optimizer"),
    n_steps: int = typer.Option(
        2048, help="Number of steps to run for each environment per update"
    ),
    batch_size: int = typer.Option(64, help="Minibatch size"),
    n_epochs: int = typer.Option(
        10, help="Number of epoch when optimizing the surrogate loss"
    ),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    n_envs: int = typer.Option(8, help="Number of parallel environments"),
    render_mode: str = typer.Option(
        None, help="Render mode for the environment (default: 'none')",
    )

):
    """Train a new model for BlueRov2 trajectory following"""
    train(
        trajectory_path=trajectory,
        output_dir=output_dir,
        total_timesteps=timesteps,
        waypoint_threshold=threshold,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        n_envs=n_envs,
        render_mode=render_mode,
    )


if __name__ == "__main__":
    app()
