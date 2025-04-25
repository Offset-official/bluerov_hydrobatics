# run_bluerov_ppo.py
import argparse
import os
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import time
from bluerov2_gym.envs.bluerov_env import BlueRov


# --------------------------------------------------------------------------- #
# Helper: build a single-env VecEnv so SB3 APIs stay happy
# --------------------------------------------------------------------------- #
def make_env(render_mode, trajectory_file):
    def _init():
        return BlueRov(
            render_mode=render_mode,  # "human" to see the sim, None for headless
            trajectory_file=trajectory_file,
        )

    return _init


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "a2c": A2C,
    "ddpg": DDPG,
}


def run_policy(
    algo: str,
    model_path: str,
    trajectory_file: str,
    episodes: int = 3,
    max_steps: int = 1000,
    render_mode: str | None = "human",
    deterministic: bool = True,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if algo not in ALGOS:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(ALGOS.keys())}")
    env = make_env(render_mode, trajectory_file)()
    ModelClass = ALGOS[algo]
    model = ModelClass.load(
        model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    episode_returns = []
    env.unwrapped.set_waypoints_visualization(trajectory_file)
    for ep in range(episodes):
        obs, _ = env.reset()
        env.render()
        print(f"\nStarting Episode {ep + 1}")
        time.sleep(10)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done and ep_len < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, _ = env.step(action)
            ep_ret += reward
            ep_len += 1
            time.sleep(0.1)
            env.unwrapped.step_sim()
        episode_returns.append(ep_ret)
        print(f"Episode {ep+1}/{episodes} – return: {ep_ret:.2f}, steps: {ep_len}")
    env.close()
    print("\nSummary:")
    print(f"  Mean return: {np.mean(episode_returns):.2f}")
    print(f"  Std  return: {np.std(episode_returns):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run a trained RL policy on BlueROV")
    p.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(ALGOS.keys()),
        help="RL algorithm to use",
    )
    p.add_argument(
        "--model", required=True, help="Path to model (e.g. runs/.../best_model.zip)"
    )
    p.add_argument(
        "--trajectory_file", required=True, help="CSV with (x,y,z) way-points"
    )
    p.add_argument("--episodes", type=int, default=3, help="How many rollouts to run")
    p.add_argument("--max_steps", type=int, default=1000, help="Step limit per episode")
    p.add_argument("--no_render", action="store_true", help="Disable visual rendering")
    p.add_argument("--stochastic", action="store_true", help="Use stochastic actions")
    args = p.parse_args()
    loaded_trajectory = np.loadtxt(args.trajectory_file, delimiter=",")
    run_policy(
        algo=args.algo,
        model_path=args.model,
        trajectory_file=loaded_trajectory,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode=None if args.no_render else "human",
        deterministic=not args.stochastic,
    )
