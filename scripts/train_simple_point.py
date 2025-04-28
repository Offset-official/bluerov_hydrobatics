import time
from pathlib import Path
import gymnasium as gym
import bluerov2_gym
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
import rich

import bluerov2_gym.envs

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    output_dir: str = "./trained_models",
    total_timesteps: int = 1000000,
    n_steps: int = 8,
    n_envs: int = 8,
    model_name: str = "bluerov_simplepoint",
    render_mode: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_name

    print(f"Model will be saved to: {model_path}")
    print(f"Using {n_envs} parallel environments")

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=model_name,
        verbose=1
    )

    eval_vec_env = VecNormalize(
        make_vec_env(
            "BlueRov-v0",
            n_envs=1,
            seed=42,
            env_kwargs={"render_mode": "none"},
        )
    )

    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(output_dir / "best_checkpoints"),
        log_path="./logs/",
        verbose=1,
        eval_freq=1000,
        deterministic=True,
        n_eval_episodes=20,
        # render=True,
    )

    vec_env = VecNormalize(
        make_vec_env(
            "BlueRov-v0",
            n_envs=n_envs,
            seed=42,
            env_kwargs={"render_mode": None},
            vec_env_cls=SubprocVecEnv,
        )
    )

    env = bluerov2_gym.envs.BlueRov(render_mode=None)

    check_env(env, warn=True, skip_render_check=True)

    print("Environment check passed successfully ✅")

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=0,
        n_steps=n_steps,
        batch_size=64,
        device="cpu",
        tensorboard_log=str("logs"),
    )

    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    app()
