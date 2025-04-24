# train_bluerov_ppo.py
import argparse
import os
from datetime import datetime
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from bluerov2_gym.envs.bluerov_env import BlueRov


# --------------------------------------------------------------------------- #
# Helper: each subprocess gets its own env instance
# --------------------------------------------------------------------------- #
def make_env(render_mode, trajectory_file):
    def _init():
        return BlueRov(
            render_mode=render_mode,
            trajectory_file=trajectory_file
        )
    return _init


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def start_ppo_training(
    trajectory_file: str,
    num_episodes: int = 1000,
    steps: int = 1000,
    n_envs: int = 8,
    render_mode_eval: str | None = "human",
):
    """
    Train a PPO policy that follows the given trajectory.

    Args
    ----
    trajectory_file : str
        CSV with way-points (x,y,z).f
    num_episodes : int
        Logical episodes – only used to scale total_timesteps = episodes*steps.
    steps : int
        Episode length in env steps.
    n_envs : int
        Number of parallel simulators (DummyVecEnv).
    render_mode_eval : str
        Render mode for the evaluation env ("human" or None).
    """

    # ---------- bookkeeping ----------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"PPO_BlueROV_{run_tag}")
    best_dir = os.path.join(out_dir, "best"); ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(best_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

    # ---------- vectorised envs ----------
    env     = DummyVecEnv([make_env(None, trajectory_file) for _ in range(n_envs)])
    evalenv = DummyVecEnv([make_env(render_mode_eval, trajectory_file)])

    # ---------- PPO agent ----------
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=2048 // n_envs,   # roll-out per env
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(out_dir, "tb"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1
    )

    # ---------- callbacks ----------
    eval_cb = EvalCallback(
        evalenv,
        n_eval_episodes=5,
        best_model_save_path=best_dir,
        eval_freq=10_000,
        log_path=os.path.join(out_dir, "eval"),
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=ckpt_dir,
        name_prefix="ppo_bluerov"
    )

    # ---------- learn ----------
    total_steps = num_episodes * steps
    model.learn(total_timesteps=total_steps,
                callback=[eval_cb, ckpt_cb],
                progress_bar=True)

    # ---------- save ----------
    model.save(os.path.join(out_dir, "final_ppo_policy"))
    env.close(); evalenv.close()
    print(f"\nTraining complete – artefacts in {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PPO on BlueROV trajectory")
    p.add_argument("--trajectory_file", type=str, required=True,
                   help="Path to trajectory CSV")
    p.add_argument("--num_episodes", type=int, default=10,
                   help="Episodes (used to compute total timesteps)")
    p.add_argument("--steps", type=int, default=10,
                   help="Max env steps per episode")
    p.add_argument("--envs", type=int, default=8,
                   help="Parallel envs")
    
    args = p.parse_args()

    trajectory = np.loadtxt(args.trajectory_file, delimiter=",")
    start_ppo_training(
        trajectory_file=trajectory,
        num_episodes=args.num_episodes,
        steps=args.steps,
        n_envs=args.envs,
    )

