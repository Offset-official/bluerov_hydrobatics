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
        current_ep_angle_offsets = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            total_reward += reward
            # current_ep_rewards.append(sum(info[0]["reward_tuple"]))
            # current_ep_reward_tuples.append(info[0]["reward_tuple"])

            # distances_from_goal.append(info[0]["distance_from_goal"])
            # current_ep_angle_offsets.append(info[0]["angle_offset"])

            time.sleep(0.1)
            env.unwrapped.env_method("step_sim")
            done = dones[0]
            if done:
                success = bool(info[0].get("is_success", False))

        episode_rewards.append(total_reward)
        success_count += success
        # print(
        #     f"Episode {ep}/{num_episodes} — Reward: {total_reward:.2f}  Success: {success} Episode Length: {len(distances_from_goal)}"
        # )
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # ax1.plot(distances_from_goal, label="Distance from goal", color="blue")
        # ax1.plot(current_ep_rewards, label="Reward", color="orange")
        # ax1.plot(current_ep_angle_offsets, label="Angle offset", color="cyan")
        # ax1.set_xlabel("Time step")
        # ax1.set_ylabel("Value")
        # ax1.set_title(f"Episode {ep} - Distance from Goal and Reward")
        # ax1.legend()
        # ax1.grid()

        # ax2.plot(
        #     [x[0] for x in current_ep_reward_tuples],
        #     label="Position reward",
        #     color="green",
        # )
        # ax2.plot(
        #     [x[1] for x in current_ep_reward_tuples], label="Angle reward", color="red"
        # )
        # ax2.plot(
        #     [x[2] for x in current_ep_reward_tuples],
        #     label="Action reward",
        #     color="purple",
        # )
        # ax2.plot(
        #     [x[3] for x in current_ep_reward_tuples],
        #     label="Completion reward",
        #     color="brown",
        # )
        # ax2.set_xlabel("Time step")
        # ax2.set_ylabel("Reward Components")
        # ax2.set_title(f"Episode {ep} - Reward Components")
        # ax2.legend()
        # ax2.grid()

        # plt.tight_layout()
        # plt.show()

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
