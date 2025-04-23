import gymnasium as gym
import bluerov2_gym

# Create the environment
env = gym.make("BlueRov-v0", render_mode="human")

# Reset the environment
observation, info = env.reset()

# Run a simple control loop
while True:
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()