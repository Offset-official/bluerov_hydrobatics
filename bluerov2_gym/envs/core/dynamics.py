import numpy as np
from scipy import stats


class Dynamics:
    """
    BlueROV2 Dynamics Model

    This class implements the physics-based dynamics model for the BlueROV2 underwater vehicle.
    It includes hydrodynamic coefficients, disturbance modeling, and state update equations.
    """

    def __init__(self):
        """
        Initialize the dynamics model with hydrodynamic parameters and disturbance model.

        Parameters:
        - a_* : Hydrodynamic coefficients for damping forces
        - b_* : Control effectiveness coefficients
        """
        # Hydrodynamic parameters based on system identification
        self.params = {
            # Linear damping coefficients
            "a_vx": -0.5265,  # X-axis linear damping
            "a_vy": -0.5357,  # Y-axis linear damping
            "a_vz": -1.0653,  # Z-axis linear damping
            "a_vw": -4.2579,  # Angular damping
            # Quadratic damping coefficients
            "a_vx2": -1.1984,  # X-axis quadratic damping
            "a_vy2": -5.1626,  # Y-axis quadratic damping
            "a_vz2": -1.7579e-5,  # Z-axis quadratic damping
            "a_vw2": -2.4791e-8,  # Angular quadratic damping
            # Coupled dynamics coefficients
            "a_vyw": -0.5350,  # Y-velocity and angular velocity coupling
            "a_vwx": -1.2633,  # X-velocity and angular velocity coupling
            "a_vxy": -0.9808,  # X-Y velocity coupling
            # Control effectiveness coefficients
            "b_vx": 1.2810,  # X-axis control effectiveness
            "b_vy": 0.9512,  # Y-axis control effectiveness
            "b_vz": 0.7820,  # Z-axis control effectiveness
            "b_vw": 2.6822,  # Angular control effectiveness
        }

        # Simulation time step
        self.dt = 0.1  # seconds

        # Environmental disturbance model (e.g., water currents, unmodeled dynamics)
        self.disturbance_mean = np.array(
            [-0.01461447, -0.02102184, -0.00115958, 0.05391866]
        )
        self.disturbance_cov = np.array(
            [
                [2.89596342e-2, 5.90296868e-3, -4.22672521e-5, -6.38837738e-3],
                [5.90296868e-3, 2.05937494e-2, 8.59805304e-5, 2.92258483e-3],
                [-4.22672521e-5, 8.59805304e-5, 2.44296056e-3, 1.64117342e-3],
                [-6.38837738e-3, 2.92258483e-3, 1.64117342e-3, 3.71338116e-1],
            ]
        )
        # Multivariate normal distribution for generating realistic disturbances
        self.disturbance_dist = stats.multivariate_normal(
            mean=self.disturbance_mean, cov=self.disturbance_cov
        )

    def step(self, state, action):
        """
        Update the state based on current state, actions, and disturbances.

        Args:
            state (dict): Current state of the BlueROV
            action (numpy.ndarray): Control inputs (normalized thruster commands)

        Returns:
            None: Updates the state dictionary in-place
        """
        # Generate random disturbances from the disturbance distribution
        disturbances = self.disturbance_dist.rvs()
        dvx, dvy, dvz, domega = disturbances

        # Extract action components
        w_x, w_y, w_z, w_omega = action

        # Extract current state variables
        x, y, z, theta = (
            state["x"],
            state["y"],
            state["z"],
            state["theta"],
        )
        vx, vy, vz, omega = (
            state["vx"],
            state["vy"],
            state["vz"],
            state["omega"],
        )

        # Position updates (kinematic equations)
        # Convert body-frame velocities to world-frame positions
        state["x"] += (vx * np.cos(theta) - vy * np.sin(theta)) * self.dt
        state["y"] += (vx * np.sin(theta) + vy * np.cos(theta)) * self.dt
        state["z"] += vz * self.dt
        state["theta"] += omega * self.dt
        state["theta"] = state["theta"] % (2 * np.pi)  - np.pi  # Normalize angle

        # Velocity updates (dynamic equations with hydrodynamic forces)

        # X-axis velocity update
        state["vx"] += (
            self.params["a_vx"] * vx  # Linear damping
            + self.params["a_vx2"] * vx * abs(vx)  # Quadratic damping
            + self.params["a_vyw"] * vy * omega  # Coupled dynamics effect
            + self.params["b_vx"] * w_x  # Control input
            + dvx  # Random disturbance
        ) * self.dt

        # Y-axis velocity update
        state["vy"] += (
            self.params["a_vy"] * vy  # Linear damping
            + self.params["a_vy2"] * vy * abs(vy)  # Quadratic damping
            + self.params["a_vwx"] * vx * omega  # Coupled dynamics effect
            + self.params["b_vy"] * w_y  # Control input
            + dvy  # Random disturbance
        ) * self.dt

        # Z-axis velocity update (depth)
        state["vz"] += (
            self.params["a_vz"] * vz  # Linear damping
            + self.params["a_vz2"] * vz * abs(vz)  # Quadratic damping
            + self.params["b_vz"] * w_z  # Control input
            + dvz  # Random disturbance
        ) * self.dt

        # Angular velocity update
        state["omega"] += (
            self.params["a_vw"] * omega  # Linear damping
            + self.params["a_vw2"] * omega * abs(omega)  # Quadratic damping
            + self.params["a_vxy"] * vx * vy  # Coupled dynamics effect
            + self.params["b_vw"] * w_omega  # Control input
            + domega  # Random disturbance
        ) * self.dt

    def reset(self):
        """
        Reset the dynamics model and generate new disturbances.

        Returns:
            stats.multivariate_normal: The distribution of disturbances
        """
        # Generate new disturbances for the next episode
        self.disturbances = self.disturbance_dist.rvs()
        return self.disturbance_dist
