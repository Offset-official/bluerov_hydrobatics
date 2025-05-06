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
    model_path: str, num_episodes: int, model_type: str, normalization_file: str = None
):
    env0 = DummyVecEnv(
        [
            lambda: gym.wrappers.TimeLimit(
                gym.make("BlueRov-v0", render_mode="human", trajectory_file=None),
                max_episode_steps=100,
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
    episode_rewards = []
    success_count = 0

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        success = False
        distances_from_goal = []
        env.render()
        time.sleep(5)
        current_ep_rewards = []
        current_ep_reward_tuples = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            total_reward += reward
            current_ep_rewards.append(reward)
            current_ep_reward_tuples.append(info[0]["reward_tuple"])
            time.sleep(0.1)
            env.unwrapped.env_method("step_sim")
            done = dones[0]
            if done:
                success = bool(info[0].get("is_success", False))

        episode_rewards.append(total_reward)
        position_rewards = [
            reward_tuple[0] for reward_tuple in current_ep_reward_tuples
        ]
        angle_rewards = [
            reward_tuple[1] for reward_tuple in current_ep_reward_tuples
        ]
        completion_rewards = [
            reward_tuple[2] for reward_tuple in current_ep_reward_tuples
        ]
        termination_rewards = [
            reward_tuple[3] for reward_tuple in current_ep_reward_tuples
        ]
        action_penalties = [
            reward_tuple[4] for reward_tuple in current_ep_reward_tuples
        ]
        dot_to_goals = [reward_tuple[5] for reward_tuple in current_ep_reward_tuples]
        step_penalties = [
            reward_tuple[6] for reward_tuple in current_ep_reward_tuples
        ]

        # plot all the reward components with label
        plt.figure(figsize=(12, 8))
        plt.plot(position_rewards, label="Position Reward")
        plt.plot(angle_rewards, label="Angle Reward")
        plt.plot(completion_rewards, label="Completion Reward")
        plt.plot(termination_rewards, label="Termination Reward")
        plt.plot(action_penalties, label="Action Penalty")
        plt.plot(dot_to_goals, label="Dot to Goal")
        plt.plot(step_penalties, label="Step Penalty")
        plt.title(f"Episode {ep} Reward Components")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.savefig(f"episode_{ep}_reward_components.png")


    mean_reward = sum(episode_rewards) / num_episodes
    success_rate = success_count / num_episodes * 100.0
    print("\n=== Evaluation Summary ===")
    print(f"Mean episode reward : {mean_reward:.2f}")
    print(
        f"Success rate         : {success_count}/{num_episodes} ({success_rate:.1f}%)"
    )

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
    args = parser.parse_args()

    evaluate(
        args.model_path, args.num_episodes, args.model_type, args.normalization_file
    )
