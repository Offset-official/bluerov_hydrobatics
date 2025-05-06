import time
from pathlib import Path
import gymnasium as gym
import bluerov2_gym
from bluerov2_gym.training.callbacks import SaveVectorNormalizationCallback
import typer
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from typing_extensions import Annotated
import rich

import bluerov2_gym.envs

app = typer.Typer(pretty_exceptions_enable=False)

MAX_EPISODE_STEPS = 50


@app.command()
def train(
    model_type: Annotated[str, typer.Argument(help="Model type: A2C, PPO")],
    output_dir: str = "./trained_models",
    total_timesteps: int = 10000000,
    n_steps: int = 8,
    n_envs: int = 16,
    model_name: str = "bluerov_simplepoint",
    render_mode: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{model_name}"
    model_path = output_dir / model_name

    print(f"Model will be saved to: {model_path}")
    print(f"Using {n_envs} parallel environments")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=model_name,
        verbose=1,
    )

    eval_vec_env = VecNormalize(
        make_vec_env(
            lambda: gym.wrappers.TimeLimit(
                gym.make("BlueRov-v0", render_mode=render_mode),
                max_episode_steps=MAX_EPISODE_STEPS,
            ),
            n_envs=1,
            seed=42,
        ),
        training=False,
    )

    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(output_dir / "best_checkpoints"),
        callback_on_new_best=SaveVectorNormalizationCallback(
            verbose=0, save_path=str(output_dir / "best_checkpoints")
        ),
        log_path="./logs/",
        verbose=1,
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=20,
        render=False,
    )

    vec_env = VecNormalize(
        make_vec_env(
            lambda: gym.wrappers.TimeLimit(
                gym.make("BlueRov-v0", render_mode=render_mode),
                max_episode_steps=MAX_EPISODE_STEPS,
            ),
            n_envs=n_envs,
            seed=42,
            vec_env_cls=SubprocVecEnv,
        )
    )

    env = bluerov2_gym.envs.BlueRov(render_mode=None)

    check_env(env, warn=True, skip_render_check=True)

    print("Environment check passed successfully ✅")

    if model_type == "PPO":
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=0,
            n_steps=n_steps,
            batch_size=64,
            device="cpu",
            tensorboard_log="logs",
        )
    elif model_type == "A2C":
        model = A2C(
            "MultiInputPolicy",
            vec_env,
            verbose=0,
            n_steps=n_steps,
            device="cuda",
            tensorboard_log="logs",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")


if __name__ == "__main__":
    app()
