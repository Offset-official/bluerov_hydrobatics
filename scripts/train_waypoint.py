import time
from pathlib import Path
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env

app = typer.Typer()


@app.command()
def train(
    output_dir: str = "./trained_models",
    total_timesteps: int = 1000000,
    n_steps: int = 5,
    n_envs: int = 8,
    model_name: str = "bluerov_simplepoint",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_name

    print(f"Model will be saved to: {model_path}")
    print(f"Using {n_envs} parallel environments")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=model_name,
    )

    vec_env = make_vec_env(
        "bluerov2:bluerov2-v0",
        n_envs=n_envs,
        seed=42,
    )

    model = PPO("MultiInputPolicy", vec_env, verbose=1, n_steps=n_steps)

    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model


app = typer.Typer(help="Train BlueRov2 on trajectory following")

if __name__ == "__main__":
    app()
