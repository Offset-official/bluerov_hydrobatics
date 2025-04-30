from importlib import resources
from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward, SinglePointReward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer
from random import random


class BlueRov(gym.Env):
    """
    BlueROV2 Gymnasium Environment

    This environment simulates the dynamics of a BlueROV2 underwater vehicle
    for reinforcement learning tasks. It includes position and velocity states
    in a 3D environment with heading angle.

    State variables:
    - x, y, z: 3D position coordinates
    - theta: heading angle
    - vx, vy, vz: linear velocities
    - omega: angular velocity
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_file=None):
        """
        Initialize the BlueROV environment

        Args:
            render_mode (str, optional): Rendering mode. Use "human" for visualization.
            trajectory_file (str, optional): Path to CSV file containing waypoint trajectory.
        """
        super().__init__()

        if render_mode is not None:
            with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
                self.model_path = str(asset_path)
            self.renderer = BlueRovRenderer()
            self.render_mode = render_mode

        self.dynamics = Dynamics()
        self.trajectory_file = trajectory_file
        self.trajectory = None
        self.threshold_distance = 0.1
        self.angular_threshold = 0.1

        self.distance_to_goal_from_start = 0.0

        init_x = 0.0
        init_y = 0.0
        init_z = 0.0
        init_theta = 0.0

        # Load trajectory if provided
        if trajectory_file is not None:
            self.trajectory = np.loadtxt(trajectory_file, delimiter=",")
            print(f"Loaded trajectory with {self.trajectory.shape[0]} waypoints")
            init_x = self.trajectory[0, 0]
            init_y = self.trajectory[0, 1]
            init_z = self.trajectory[0, 2]
            init_theta = self.trajectory[0, 3]
            self.goal_point = self.trajectory[1, :]
        else:
            self.trajectory = None
            self.goal_point, self.distance_to_goal_from_start = (
                self.compute_random_goal_point()
            )
        self.waypoint_idx = 1
        self.reward_fn = SinglePointReward(
            threshold=self.threshold_distance, angular_threshold=self.angular_threshold
        )

        self.number_of_steps = 0

        self.state = {
            "x": init_x,  # x position (m)
            "y": init_y,  # y position (m)
            "z": init_z,  # depth (m)
            "theta": init_theta,  # heading angle (rad)
            "vx": 0.0,  # x velocity (m/s)
            "vy": 0.0,  # y velocity (m/s)
            "vz": 0.0,  # vertical velocity (m/s)
            "omega": 0.0,  # angular velocity (rad/s)
        }

        if self.trajectory is not None:
            self.distance_to_goal_from_start = self.compute_distance_from_goal()

        self.init_state = deepcopy(self.state)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "offset_x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "offset_y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "offset_z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "offset_theta": spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float64
                ),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )

        self.dt = 0.1  # Time step (seconds)
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.render()

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional configuration options

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        spiral_traj = """
        5.0,0.0,0.0,0.09519977738150898
4.909643486313533,0.9462562218020509,-0.09090909090909091,0.2855993321445265
4.641839665080363,1.8583122783016375,-0.18181818181818182,0.4759988869075449
4.206267664155906,2.703204087277988,-0.2727272727272727,0.6663984416705624
3.618670190525351,3.4503950574105597,-0.36363636363636365,0.8567979964335799
2.9002845478559913,4.072879760251678,-0.4545454545454546,1.0471975511965974
2.0770750650094323,4.548159976772592,-0.5454545454545454,1.2375971059596158
1.1787946775471365,4.859057841617708,-0.6363636363636364,1.4279966607226329
0.23790957911871202,4.99433669591504,-0.7272727272727273,-4.664789091693936
-0.7115741913664251,4.949107209404664,-0.8181818181818182,-4.474389536930918
-1.6353398165871071,4.725004093573343,-0.9090909090909092,-4.283989982167899
-2.499999999999999,4.330127018922194,-1.0,-4.093590427404882
-3.274303669726425,3.7787478717712912,-1.0909090909090908,-3.903190872641865
-3.9302654737139364,3.0907949311030274,-1.1818181818181819,-3.7127913178788465
-4.444177243274617,2.2911326086370525,-1.2727272727272727,-3.522391763115829
-4.7974648680724865,1.4086627842071504,-1.3636363636363638,-3.331992208352811
-4.977359612865423,0.4752802165209144,-1.4545454545454546,-3.141592653589793
-4.977359612865423,-0.4752802165209131,-1.5454545454545454,-2.951193098826775
-4.797464868072487,-1.4086627842071469,-1.6363636363636365,-2.760793544063757
-4.444177243274617,-2.2911326086370516,-1.7272727272727273,-2.5703939893007406
-3.9302654737139386,-3.0907949311030247,-1.8181818181818183,-2.379994434537722
-3.274303669726426,-3.778747871771291,-1.9090909090909092,-2.1895948797747047
-2.500000000000002,-4.330127018922192,-2.0,-1.9991953250116863
-1.6353398165871094,-4.725004093573341,-2.090909090909091,-1.8087957702486692
-0.7115741913664262,-4.949107209404663,-2.1818181818181817,-1.6183962154856517
0.2379095791187119,-4.99433669591504,-2.272727272727273,-1.427996660722634
1.178794677547133,-4.85905784161771,-2.3636363636363638,-1.2375971059596154
2.07707506500943,-4.548159976772593,-2.4545454545454546,-1.0471975511965976
2.9002845478559895,-4.072879760251679,-2.5454545454545454,-0.8567979964335806
3.6186701905253504,-3.45039505741056,-2.6363636363636362,-0.6663984416705628
4.206267664155904,-2.703204087277991,-2.7272727272727275,-0.47599888690754444
4.6418396650803615,-1.8583122783016404,-2.8181818181818183,-0.28559933214452693
4.9096434863135325,-0.9462562218020532,-2.909090909090909,-0.09519977738150964
5.0,-1.2246467991473533e-15,-3.0,0.09519977738150898
4.909643486313533,0.9462562218020507,-3.090909090909091,0.2855993321445265
4.641839665080364,1.858312278301634,-3.181818181818182,0.4759988869075431
4.206267664155908,2.7032040872779857,-3.272727272727273,0.666398441670562
3.618670190525352,3.4503950574105584,-3.3636363636363638,0.8567979964335795
2.9002845478559918,4.072879760251678,-3.4545454545454546,1.0471975511965974
2.0770750650094363,4.54815997677259,-3.5454545454545454,1.2375971059596145
1.1787946775471398,4.859057841617708,-3.6363636363636367,1.4279966607226329
0.23790957911871435,4.99433669591504,-3.7272727272727275,-4.664789091693936
-0.7115741913664237,4.949107209404664,-3.8181818181818183,-4.474389536930918
-1.6353398165871071,4.725004093573343,-3.909090909090909,-4.2839899821679
-2.499999999999996,4.330127018922195,-4.0,-4.093590427404882
-3.274303669726426,3.778747871771291,-4.090909090909091,-3.903190872641865
-3.930265473713936,3.0907949311030283,-4.181818181818182,-3.712791317878847
-4.4441772432746145,2.2911326086370574,-4.2727272727272725,-3.5223917631158295
-4.7974648680724865,1.4086627842071495,-4.363636363636363,-3.331992208352811
-4.977359612865422,0.4752802165209178,-4.454545454545455,-3.141592653589794
-4.977359612865423,-0.4752802165209141,-4.545454545454546,-2.951193098826776
-4.797464868072488,-1.408662784207146,-4.636363636363637,-2.7607935440637577
-4.44417724327462,-2.2911326086370467,-4.7272727272727275,-2.570393989300741
-3.930265473713938,-3.090794931103025,-4.818181818181818,-2.3799944345377226
-3.2743036697264287,-3.7787478717712886,-4.909090909090909,-2.189594879774705
-2.5000000000000067,-4.330127018922189,-5.0,-1.9991953250116874
-1.6353398165871105,-4.725004093573341,-5.090909090909091,-1.8087957702486703
-0.7115741913664319,-4.949107209404663,-5.181818181818182,-1.6183962154856513
0.2379095791187107,-4.99433669591504,-5.2727272727272725,-1.427996660722634
1.178794677547132,-4.85905784161771,-5.363636363636364,-1.2375971059596167
2.077075065009425,-4.548159976772595,-5.454545454545455,-1.0471975511965983
2.900284547855989,-4.07287976025168,-5.545454545454546,-0.8567979964335806
3.6186701905253464,-3.450395057410564,-5.636363636363637,-0.6663984416705633
4.206267664155906,-2.7032040872779883,-5.7272727272727275,-0.47599888690754466
4.6418396650803615,-1.8583122783016415,-5.818181818181818,-0.2855993321445274
4.909643486313532,-0.9462562218020587,-5.909090909090909,-0.09519977738151031
5.0,-2.4492935982947065e-15,-6.0,0.09519977738150853
4.909643486313534,0.9462562218020452,-6.090909090909091,0.2855993321445258
4.641839665080363,1.858312278301637,-6.181818181818182,0.47599888690754355
4.2062676641559085,2.7032040872779843,-6.2727272727272725,0.6663984416705611
3.618670190525356,3.450395057410555,-6.363636363636364,0.8567979964335795
2.900284547855993,4.072879760251677,-6.454545454545455,1.0471975511965974
2.0770750650094376,4.548159976772589,-6.545454545454546,1.237597105959615
1.1787946775471367,4.859057841617708,-6.636363636363637,1.4279966607226324
0.23790957911871558,4.99433669591504,-6.7272727272727275,-4.664789091693937
-0.7115741913664181,4.949107209404665,-6.818181818181818,-4.474389536930918
-1.635339816587106,4.725004093573343,-6.909090909090909,-4.283989982167901
-2.499999999999995,4.330127018922196,-7.0,-4.093590427404884
-3.274303669726418,3.7787478717712975,-7.090909090909091,-3.903190872641866
-3.9302654737139346,3.090794931103029,-7.181818181818182,-3.7127913178788483
-4.4441772432746145,2.2911326086370587,-7.272727272727273,-3.522391763115829
-4.7974648680724865,1.4086627842071506,-7.363636363636364,-3.331992208352811
-4.977359612865422,0.47528021652091906,-7.454545454545455,-3.141592653589795
-4.977359612865424,-0.4752802165209041,-7.545454545454546,-2.951193098826777
-4.797464868072488,-1.4086627842071446,-7.636363636363637,-2.7607935440637585
-4.444177243274621,-2.2911326086370454,-7.7272727272727275,-2.5703939893007406
-3.9302654737139386,-3.0907949311030243,-7.818181818181818,-2.3799944345377235
-3.274303669726436,-3.778747871771282,-7.909090909090909,-2.189594879774707
-2.500000000000008,-4.330127018922188,-8.0,-1.9991953250116883
-1.6353398165871118,-4.725004093573341,-8.090909090909092,-1.808795770248669
-0.7115741913664242,-4.949107209404664,-8.181818181818182,-1.6183962154856513
0.2379095791187006,-4.99433669591504,-8.272727272727273,-1.4279966607226353
1.1787946775471307,-4.85905784161771,-8.363636363636363,-1.2375971059596158
2.077075065009432,-4.548159976772592,-8.454545454545455,-1.0471975511965992
2.9002845478559807,-4.072879760251686,-8.545454545454545,-0.8567979964335818
3.618670190525346,-3.450395057410565,-8.636363636363637,-0.6663984416705632
4.206267664155905,-2.7032040872779897,-8.727272727272727,-0.475998886907546
4.641839665080358,-1.8583122783016508,-8.818181818181818,-0.2855993321445285
4.909643486313532,-0.9462562218020598,-8.90909090909091,-0.09519977738151031
5.0,-3.673940397442059e-15,-9.0,-0.09519977738151031
"""     
        spiral_arr  = np.array(
            [
                [float(coord) for coord in line.split(",")]
                for line in spiral_traj.strip().split("\n")
            ]
        )
        idx = np.random.randint(0, spiral_arr.shape[0] - 1)


        # self.state = deepcopy(self.init_state)
        self.state = {
            "x": spiral_arr[idx, 0],
            "y": spiral_arr[idx, 1],
            "z": spiral_arr[idx, 2],
            "theta": spiral_arr[idx, 3],
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "omega": 0.0,
        }

        self.number_of_steps = 0

        # if self.trajectory is not None:
        #     self.waypoint_idx = 1
        #     self.goal_point = self.trajectory[self.waypoint_idx, :]
        # else:
        #     self.goal_point, self.distance_to_goal_from_start = (
        #         self.compute_random_goal_point()
        #     )

        self.goal_point = spiral_arr[idx + 1, :]
        self.distance_to_goal_from_start = self.compute_distance_from_goal()

        self.disturbance_dist = self.dynamics.reset()

        obs = self.compute_observation()

        info = {
            "distance_from_goal": self.compute_distance_from_goal(),
            "current_heading": self.state["theta"],
        }

        return obs, info

    def step(self, action):
        """
        Take a step in the environment given an action.

        Args:
            action (numpy.ndarray): Array of 4 thruster commands (-1.0 to 1.0)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.dynamics.step(self.state, action)

        self.number_of_steps += 1

        obs = self.compute_observation()

        terminated = False
        truncated = False

        # Check boundary conditions for termination
        if abs(self.state["z"]) > 10.0:  # Depth limit
            terminated = True
        if (
            abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0
        ):  # Horizontal boundaries
            terminated = True

        distance_from_goal = self.compute_distance_from_goal()

        if distance_from_goal > self.distance_to_goal_from_start + 0.5:
            terminated = True

        action_magnitude = self.compute_action_magnitude(action)

        is_success = bool(
            distance_from_goal < self.threshold_distance
            and (abs(obs["offset_theta"][0]) < self.angular_threshold)
        )

        terminated = bool(terminated or is_success)

        if is_success and self.trajectory is not None:
            self.waypoint_idx += 1
            self.goal_point = self.trajectory[self.waypoint_idx, :]
            self.distance_to_goal_from_start = self.compute_distance_from_goal()
            terminated = False

        reward_tuple = self.reward_fn.get_reward(
            distance_from_goal,
            obs["offset_theta"][0],
            action_magnitude,
            self.number_of_steps,
        )
        prev_distance_from_goal = distance_from_goal
        total_reward = reward_tuple[0]

        info = {
            "distance_from_goal": distance_from_goal,
            "reward_tuple": reward_tuple,
            "reward": total_reward,
            "action_magnitude": action_magnitude,
            "is_success": is_success,
            "angle_offset": abs(obs["offset_theta"][0]),
            "current_heading": self.state["theta"],
        }

        if self.render_mode == "human":
            self.step_sim()

        return obs, total_reward, terminated, truncated, info

    def render(self):
        """
        Render the environment if in human mode.
        """
        self.renderer.render(self.model_path, self.init_state)
        if self.trajectory is not None:
            self.renderer.visualize_waypoints(
                self.trajectory,
                current_idx=self.waypoint_idx,
            )
        else:
            self.renderer.visualize_waypoints(
                [[0, 0, 0, 0], self.goal_point],
                current_idx=1,
            )

    def step_sim(self):
        """
        Update the visualization with the current state.
        """
        self.renderer.step_sim(self.state)

        if self.trajectory is not None:
            self.renderer.visualize_waypoints(
                self.trajectory,
                current_idx=self.waypoint_idx,
            )
        else:
            self.renderer.visualize_waypoints(
                [[0, 0, 0, 0], self.goal_point],
                current_idx=1,
            )

    def compute_observation(self):

        obs = {
            "offset_x": np.array([self.state["x"] - self.goal_point[0]]),
            "offset_y": np.array([self.state["y"] - self.goal_point[1]]),
            "offset_z": np.array([self.state["z"] - self.goal_point[2]]),
            "offset_theta": np.array([self.state["theta"] - self.goal_point[3]]),
            "vx": np.array([self.state["vx"]]),
            "vy": np.array([self.state["vy"]]),
            "vz": np.array([self.state["vz"]]),
            "omega": np.array([self.state["omega"]]),
        }

        return obs

    def compute_distance_from_goal(self):

        return np.linalg.norm(
            np.array(
                [
                    self.state["x"] - self.goal_point[0],
                    self.state["y"] - self.goal_point[1],
                    self.state["z"] - self.goal_point[2],
                ]
            )
        )

    def compute_action_magnitude(self, action):
        return np.linalg.norm(action)

    def compute_random_goal_point(self):
        """
        Generate a random point anywhere within a sphere of radius R around the origin.
        """

 
        # # Parse the trajectory string into a list of floats
        # trajectory = [
        #     [float(coord) for coord in line.split(",")]
        #     for line in spiral_traj.strip().split("\n")
        # ]
        # trajectory = np.array(trajectory)
        # # take random 2 adjacent points
        # idx = np.random.randint(0, trajectory.shape[0] - 1)
        # p1 = trajectory[idx]
        # p2 = trajectory[idx + 1]
        # x = p2[0] - p1[0]
        # y = p2[1] - p1[1]
        # z = p2[2] - p1[2]
        # heading_theta = p1[]
        # R = 2
        # theta = 2 * np.pi * random()
        # phi = np.arccos(1 - 2 * random())
        # r = R * (random() ** (1 / 3))  # Cube root for uniform distribution in volume

        # x = r * np.sin(phi) * np.cos(theta)
        # y = r * np.sin(phi) * np.sin(theta)
        # z = r * np.cos(phi)

        # heading_theta = np.random.uniform(
        #     -np.pi / 2, np.pi / 2
        # )  # do not ever make the vehicle move more than 180 degrees

        # return np.array([x, y, z, heading_theta]), r
        pass
