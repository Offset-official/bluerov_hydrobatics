# TRAINING.md

# RL Policy Evaluation for BlueROV

This project supports running and evaluating trained RL policies for BlueROV using five different algorithms:

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

## Usage

Use the `runners/run.py` script to evaluate a trained model. You must specify the algorithm, model path, and trajectory CSV file. Additional options allow you to control the number of episodes, step limits, rendering, and action determinism.

### Command-line Arguments

- `--algo`           : RL algorithm to use (`ppo`, `sac`, `td3`, `a2c`, `ddpg`)
- `--model`          : Path to the trained model (e.g. `runs/.../best_model.zip`)
- `--trajectory_file`: Path to CSV file with waypoints (x, y, z)
- `--episodes`       : Number of rollouts to run (default: 3)
- `--max_steps`      : Step limit per episode (default: 1000)
- `--no_render`      : Disable visual rendering (optional)
- `--stochastic`     : Use stochastic actions (optional)

### Example Commands

**PPO:**
```bash
python runners/run.py --algo ppo --model runs/PPO_BlueROV_20250423_202746/best/best_model.zip --trajectory_file trajectories/square.csv
```

**SAC:**
```bash
python runners/run.py --algo sac --model runs/SAC_BlueROV_20250423_202746/best/best_model.zip --trajectory_file trajectories/lemniscate.csv
```

**TD3:**
```bash
python runners/run.py --algo td3 --model runs/TD3_BlueROV_20250423_202746/best/best_model.zip --trajectory_file trajectories/spiral.csv
```

**A2C:**
```bash
python runners/run.py --algo a2c --model runs/A2C_BlueROV_20250423_202746/best/best_model.zip --trajectory_file trajectories/flat_square.csv
```

**DDPG:**
```bash
python runners/run.py --algo ddpg --model runs/DDPG_BlueROV_20250423_202746/best/best_model.zip --trajectory_file trajectories/flat_lemniscate.csv
```

### Notes
- The model must be trained and saved using the corresponding algorithm.
- The trajectory CSV should have columns: `x,y,z` (no header required).
- Rendering can be disabled for headless evaluation with `--no_render`.
- Use `--stochastic` to sample stochastic actions (if supported by the algorithm).

## Custom Training Scripts

To train your own models, you need to implement a trainer script for each algorithm. Use `trainers/PPO.py` as a reference and create your own `trainers/<algorithm>.py` (e.g., `trainers/SAC.py`, `trainers/TD3.py`, etc.) for other algorithms.

## Customizing Rewards

To experiment with different reward functions, modify the `get_reward_trajectory` function in the `Reward` class of `BlueRovEnv`. This function accepts an `np.array` and determines the reward logic for the agent. Adjusting this function allows you to test and optimize different reward strategies for your experiments.
