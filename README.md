# BlueROV2 Hydrobatics 🌊 

Reinforcement Learning project to perform complex maneuvers using the BlueROV2.

## Setup

```bash
# Clone the repository
git clone https://github.com/Offset-official/bluerov_hydrobatics
cd bluerov2_gym

# Create and activate a virtual environment
python -m venv br2gym.venv
source br2gym.venv/bin/activate

# Install the package and dependencies
pip install -e .
```

## Trajectory Types

- Spiral Trajectory

- Lemniscate (Figure-Eight) Trajectory

- Square Trajectory

## ️Quick Start


### Generating Trajectories

```bash
# Generate a spiral trajectory
python scripts/generate_trajectory.py --type spiral --depth 9.0 --points 100 --plot

# Generate a lemniscate (figure-eight) trajectory
python scripts/generate_trajectory.py --type lemniscate --depth 5.0 --points 150 --plot

# Generate a square trajectory and save to a specific file
python scripts/generate_trajectory.py --type square --depth 7.0 --points 200 --output my_square_trajectory.csv
```

### Visualizing Trajectories

```bash
# Visualize a trajectory with the BlueROV2 model
python scripts/visualize_trajectory.py trajectories/spiral.csv

# Loop animation with increased speed
python scripts/visualize_trajectory.py trajectories/lemniscate.csv --loop --speed 2.0

# Visualize one of your generated trajectories
python scripts/visualize_trajectory.py my_square_trajectory.csv
```


### Main Algorithm Runner


```bash
# Run PID controller with a trajectory file with max steps = 200000
python ./scripts/main.py --algorithm pid --file ./trajectories/spiral.csv --max-steps 200000

# Run manual control
python ./scripts/main.py --algorithm manual


python scripts/train_simple_point.py --model-type sac --total-timesteps 500000 --n-envs 4 --model-name mymodel

python ./scripts/evaluate_waypoint_model.py --trajectory-file ./trajectories/spiral.csv --model-type a2c --model-path ./trained_models/best_model.zip --normalization-file ./trained_models/best_vector_norm.pkl
```