import time
from pathlib import Path
import gymnasium as gym
import bluerov2_gym
import typer
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
import rich

import bluerov2_gym.envs

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    output_dir: str = "./trained_models",
    total_timesteps: int = 1000000,
    n_steps: int = 8,
    n_envs: int = 8,
    render_mode: str = None,
    best_save_folder = "best_checkpoints",
):

    env = bluerov2_gym.envs.BlueRov(render_mode=None)
    check_env(env, warn=True, skip_render_check=True)
    print("Environment check passed successfully ✅")


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update model_name to include model_type for clarity
    # model_name = f"{model_type.lower()}_{model_name}"
    # model_path = output_dir / model_name
    best_path = output_dir / best_save_folder

    print(f"Model will be saved to: ", best_path)
    print(f"Using {n_envs} parallel environments")


    train_env = VecNormalize(
        make_vec_env(
            "BlueRov-v0",
            n_envs=n_envs,
            seed=42,
            env_kwargs={"render_mode": None},
            vec_env_cls=SubprocVecEnv,
        )
    )

    class FrozenEvalCallback(EvalCallback):
        def __init__(
            self,
            train_env: VecNormalize,
            best_model_save_path: str,
            log_path: str = None,
            eval_freq: int = 1_000,
            n_eval_episodes: int = 20,
            deterministic: bool = True,
            verbose: int = 1,
            callback_on_new_best: BaseCallback | None = None,
        ):

            frozen_eval = VecNormalize(
                make_vec_env(
                    "BlueRov-v0",
                    n_envs=1,
                    seed=42,
                    env_kwargs={"render_mode": None},
                    vec_env_cls=DummyVecEnv,
                )
                , training=False, norm_reward=False)

            frozen_eval.obs_rms = train_env.obs_rms
            frozen_eval.ret_rms = train_env.ret_rms

            super().__init__(
                eval_env=frozen_eval,
                best_model_save_path=best_model_save_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                verbose=verbose,
                callback_on_new_best=callback_on_new_best,
            )

    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(
            self,
            vecnormalize_env: VecNormalize,
            save_path: str | Path,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            if not isinstance(vecnormalize_env, VecNormalize):
                raise TypeError(
                    "SaveVecNormalizeCallback expects a `VecNormalize` env, "
                    f"got {type(vecnormalize_env)}"
                )

            self.vecnormalize_env = vecnormalize_env
            self.save_path = Path(save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        def _on_step(self) -> bool:
            self.vecnormalize_env.save(self.save_path)
            if self.verbose > 0:
                print(f"[SaveVecNormalizeCallback] VecNormalize stats saved → {self.save_path}")
            return True

    new_best_eval_callback = SaveVecNormalizeCallback(
        vecnormalize_env=train_env,           # your *training* VecNormalize wrapper
        save_path=best_path / "vecnormalize.pkl",
        verbose=1,
    )



    eval_callback = FrozenEvalCallback(
        train_env,
        best_model_save_path=str(best_path),
        log_path="./logs/",
        eval_freq=1_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
        callback_on_new_best=new_best_eval_callback
    )


    # model = PPO(
    #     "MultiInputPolicy",
    #     vec_env,
    #     verbose=0,
    #     n_steps=n_steps,
    #     batch_size=64,
    #     device="cuda",
    #     tensorboard_log=str("logs"),
    # )

    model = A2C(
        "MultiInputPolicy",
        train_env,
        verbose=0,
        n_steps=n_steps,
        # batch_size=64,
        device="cuda",
        tensorboard_log=str("logs"),
    )

    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model saved to {model_path}")
    
    # del model
    # # load best model 
    # model = A2C.load(str(best_path))
    # vec_norm = model.get_vec_normalize()

    del model 



    return model


if __name__ == "__main__":
    app()
