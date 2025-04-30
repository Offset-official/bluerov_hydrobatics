import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import bluerov2_gym
import time
import os


class BlueRovRenderer:

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.vis = meshcat.Visualizer()
        self.vis.open()

    def render(self, model_path, init_state):
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
        self.vis["vessel"].set_object(
            g.DaeMeshGeometry.from_file(model_path),
            g.MeshLambertMaterial(
                map=g.ImageTexture(
                    image=g.PngImage.from_file(
                        os.path.join(
                            bluerov2_gym.__path__[0],
                            "assets",
                            "texture.png",
                        )
                    )
                )
            ),
        )
        # cheeky fix to position the model correctly
        self.step_sim(init_state)

        ground = g.Box([30, 30, 0.01])
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -10])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)

        # Add a reference frame
        self.vis["reference_frame"].set_object(g.triad(1.0))

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
            waypoints: List of waypoints as [x, y, z, theta] coordinates
            current_idx: Index of the current target waypoint
        """
        if self.render_mode != "human":
            return

        for i, point in enumerate(waypoints):

            # arrow visualization
            length = 0.2
            cylinder_length = length * 0.7  # Cylinder part of the arrow
            cone_length = length * 0.3  # Cone tip of the arrow
            # Create cylinder for arrow shaft
            cylinder_vis = g.Cylinder(height=cylinder_length, radius=0.005)
            cylinder_material = g.MeshPhongMaterial(color=0x30C5FF)
            # Create cone for arrow tip
            cone_vis = g.Cylinder(
                height=cone_length, radius=0.02, radiusTop=0.0, radiusBottom=0.03
            )
            cone_material = g.MeshPhongMaterial(color=0x30C5FF)
            position = point[0:3]
            yaw_angle = point[3]
            # Create transforms
            initial_rotation = tf.rotation_matrix(-np.pi / 2, [0, 1, 0])
            yaw_rotation = tf.rotation_matrix(yaw_angle, [0, 0, 1])
            # Position the cylinder (shaft)
            cylinder_offset = tf.translation_matrix(
                [0, +length / 4 + cylinder_length / 2, 0]
            )
            cylinder_transform = tf.concatenate_matrices(
                tf.translation_matrix(position),
                yaw_rotation,
                initial_rotation,
                cylinder_offset,
            )

            # Position the cone (tip)
            cone_offset = tf.translation_matrix([0, length, 0])
            cone_transform = tf.concatenate_matrices(
                tf.translation_matrix(position),
                yaw_rotation,
                initial_rotation,
                cone_offset,
            )

            # Add to visualizer
            arrow_path = f"waypoints/arrow_{i}"
            self.vis[f"{arrow_path}/shaft"].set_object(cylinder_vis, cylinder_material)
            self.vis[f"{arrow_path}/shaft"].set_transform(cylinder_transform)
            self.vis[f"{arrow_path}/tip"].set_object(cone_vis, cone_material)
            self.vis[f"{arrow_path}/tip"].set_transform(cone_transform)

            # Visualize waypoints as spheres
            sphere = g.Sphere(0.1)

            if i < current_idx:
                # Past waypoints (reached) - green
                material = g.MeshPhongMaterial(color=0x00FF00)
            elif i == current_idx:
                # Current waypoint - yellow sphere
                material = g.MeshPhongMaterial(color=0xFFFF00)
            else:
                # Future waypoints - white
                material = g.MeshPhongMaterial(color=0xFFFFFF)

            self.vis[f"waypoints/point_{i}"].set_object(sphere, material)
            self.vis[f"waypoints/point_{i}"].set_transform(
                tf.translation_matrix(point[0:3])
            )
