import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

import bluerov2_gym


class BlueRovRenderer:

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.vis = meshcat.Visualizer()
        self.vis.open()

    def render(self, model_path):
        if self.render_mode != "human":
            return
        water_surface = g.Box([30, 30, 0.01])
        water_material = g.MeshPhongMaterial(
            color=0x2389DA, opacity=0.3, transparent=True, side="DoubleSide"
        )
        self.vis["water_surface"].set_object(water_surface, water_material)

        water_volume = g.Box([30, 30, -50])
        water_volume_material = g.MeshPhongMaterial(
            color=0x1A6B9F, opacity=0.2, transparent=True
        )
        water_volume_transform = tf.translation_matrix([0, 0, -5])
        self.vis["water_volume"].set_object(water_volume, water_volume_material)
        self.vis["water_volume"].set_transform(water_volume_transform)
        print("model_path: ", model_path)
        self.vis["vessel"].set_object(
            g.DaeMeshGeometry.from_file(model_path),
            g.MeshLambertMaterial(color=0x0000FF, wireframe=False),
        )

        ground = g.Box([30, 30, 0.01])
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -10])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)

        # Add a reference frame
        self.vis["reference_frame"].set_object(g.TriadGeometry(1.0))

    def step_sim(self, state):
        self.state = state  # maybe wrong. check later
        if self.render_mode != "human":
            return

        translation = np.array([self.state["x"], self.state["y"], self.state["z"]])
        rotation_matrix = np.array(
            [
                [np.cos(self.state["theta"]), -np.sin(self.state["theta"]), 0],
                [np.sin(self.state["theta"]), np.cos(self.state["theta"]), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        self.vis["vessel"].set_transform(transform_matrix)

    def visualize_waypoints(self, waypoints, current_idx=0):
        """
        Visualize trajectory waypoints in the environment

        Args:
            waypoints: List of waypoints as [x, y, z] coordinates
            current_idx: Index of the current target waypoint
        """
        if self.render_mode != "human":
            return

        # Clear previous waypoints
        self.vis["waypoints"].delete()

        for i, point in enumerate(waypoints):
            if i < current_idx:
                # Past waypoints (reached) - small green
                sphere = g.Sphere(0.1)
                material = g.MeshPhongMaterial(color=0x00FF00)
            elif i == current_idx:
                # Current waypoint - larger yellow
                sphere = g.Sphere(0.3)
                material = g.MeshPhongMaterial(color=0xFFFF00)
            else:
                # Future waypoints - small white
                sphere = g.Sphere(0.1)
                material = g.MeshPhongMaterial(color=0xFFFFFF)

            self.vis[f"waypoints/point_{i}"].set_object(sphere, material)
            self.vis[f"waypoints/point_{i}"].set_transform(tf.translation_matrix(point))
