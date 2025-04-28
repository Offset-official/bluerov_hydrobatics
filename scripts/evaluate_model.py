#!/usr/bin/env python3
import time
import argparse

import gymnasium as gym
import bluerov2_gym             # ensure custom env is registered
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def evaluate(model_path: str, num_episodes: int):
    # Create a single env with MeshCat rendering enabled
    env = gym.make("BlueRov-v0", render_mode="human")
    model = PPO.load(model_path)

    episode_rewards = []
    success_count = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        success = False
        env.render()
        distances_from_goal = []
        time.sleep(10)
        current_ep_rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            current_ep_rewards.append(reward)

            distances_from_goal.append(info["distance_from_goal"])
            print(f"Distance from goal: {info['distance_from_goal']:.2f}")

            # Render and slow down for visibility
            
            time.sleep(0.1)
            env.unwrapped.step_sim()
            done = terminated or truncated
            if done:
                success = bool(info.get("is_success", False))

        episode_rewards.append(total_reward)
        success_count += success
        print(f"Episode {ep}/{num_episodes} — Reward: {total_reward:.2f}  Success: {success}")
        plt.plot(distances_from_goal, label="Distance from goal")
        plt.plot(current_ep_rewards, label="Reward")
        plt.xlabel("Time step")
        # plt.ylabel("Distance from goal")
        plt.title(f"Episode {ep}")
        plt.legend()
        plt.grid()
        plt.show()


    # Summary
    mean_reward = sum(episode_rewards) / num_episodes
    success_rate = success_count / num_episodes * 100.0
    print("\n=== Evaluation Summary ===")
    print(f"Mean episode reward : {mean_reward:.2f}")
    print(f"Success rate         : {success_count}/{num_episodes} ({success_rate:.1f}%)")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO model on BlueRov-v0 and report metrics"
    )
    parser.add_argument(
        "--model_path",
        help="Path to the saved model (e.g. ./trained_models/bluerov_simplepoint.zip)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation"
    )
    args = parser.parse_args()

    evaluate(args.model_path, args.num_episodes)
