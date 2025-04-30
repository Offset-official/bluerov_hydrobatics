from pathlib import Path
import csv
import time
import typer
import numpy as np
from importlib import resources
from typing import Optional
import os
import bluerov2_gym
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

app = typer.Typer(help="Vizualize a 3D trajectory with BlueROV2 model in Meshcat")


def load_trajectory_from_csv(file_path):
    """
    Load trajectory waypoints from a CSV file

    Args:
        file_path: Path to the CSV file containing x, y, z, coordinates and heading angle in rad

    Returns:
        numpy array of shape (num_points, 4) containing x, y, z, heading (rad)
    """
    waypoints = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        for row in reader:
            if len(row) >= 4:
                try:
                    x, y, z, theta = (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                    )
                    waypoints.append([x, y, z, theta])
                except ValueError:
                    print(f"Warning: Skipping invalid row {row}")

    return np.array(waypoints)


class TrajectoryVisualizer:
    """Visualize a 3D trajectory using the BlueROV2 model in Meshcat"""

    def __init__(self):
        self.vis = meshcat.Visualizer()
        self.vis.open()
        print(
            "Meshcat server started. Open the URL below in your browser if it doesn't open automatically."
        )

    def setup_scene(self, model_path):
        """Set up the underwater scene similar to BlueROV environment"""
        # Water surface
        water_surface = g.Box([30, 30, 0.01])
        water_material = g.MeshPhongMaterial(
            color=0x2389DA, opacity=0.3, transparent=True, side="DoubleSide"
        )
        self.vis["water_surface"].set_object(water_surface, water_material)

        # Water volume
        water_volume = g.Box([30, 30, -50])
        water_volume_material = g.MeshPhongMaterial(
            color=0x1A6B9F, opacity=0.2, transparent=True
        )
        water_volume_transform = tf.translation_matrix([0, 0, -5])
        self.vis["water_volume"].set_object(water_volume, water_volume_material)
        self.vis["water_volume"].set_transform(water_volume_transform)

        # BlueROV2 model
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

        # Ground/seafloor
        ground = g.Box([30, 30, 0.01])
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -10])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)

        # Add trajectory visualization
        self.vis["trajectory"].set_object(
            g.LineSegments(
                g.PointsGeometry(position=np.zeros((0, 3))),
                g.LineBasicMaterial(color=0xFF0000),
            )
        )

    def set_vessel_pose(self, position):
        """Set the position and orientation of the vessel"""
        translation = np.array(position[:3])
        theta = position[3]
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        self.vis["vessel"].set_transform(transform_matrix)

    def visualize_trajectory_points(self, trajectory):
        """Visualize trajectory as small white markers"""
        # only use x, y, z coordinates for visualization
        trajectory = trajectory[:, :3]
        points = np.array(trajectory).reshape(-1, 3)

        if len(points) > 1:
            # Add white markers for each waypoint
            for i, point in enumerate(points):
                # Special markers for start and end points
                if i == 0:
                    # Start marker (green)
                    self.vis["markers/start"].set_object(
                        g.Sphere(0.2), g.MeshPhongMaterial(color=0x00FF00)
                    )
                    self.vis["markers/start"].set_transform(
                        tf.translation_matrix(point)
                    )
                elif i == len(points) - 1:
                    # End marker (red)
                    self.vis["markers/end"].set_object(
                        g.Sphere(0.2), g.MeshPhongMaterial(color=0xFF0000)
                    )
                    self.vis["markers/end"].set_transform(tf.translation_matrix(point))
                else:
                    # Small white marker for intermediate waypoints
                    self.vis[f"markers/waypoint_{i}"].set_object(
                        g.Sphere(0.05), g.MeshPhongMaterial(color=0xFFFFFF)
                    )
                    self.vis[f"markers/waypoint_{i}"].set_transform(
                        tf.translation_matrix(point)
                    )

    def animate_trajectory(self, trajectory, loop=False, speed=1.0):
        """
        Animate the vessel moving along the trajectory

        Args:
            trajectory: numpy array of shape (num_points, 4) with x, y, z, heading coordinates
            loop: Whether to loop the animation continuously
            speed: Speed factor for the animation (higher is faster)
        """
        num_points = trajectory.shape[0]
        if num_points < 2:
            print("Trajectory needs at least 2 points for animation")
            return

        # Visualize the full trajectory path
        self.visualize_trajectory_points(trajectory)

        print(f"Animating trajectory with {num_points} waypoints...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                for i in range(num_points - 1):
                    current_pos = trajectory[i]
                    next_pos = trajectory[i + 1]

                    # Update vessel position
                    self.set_vessel_pose(current_pos)

                    # Delay based on speed
                    time.sleep(0.1 / speed)

                if not loop:
                    break

        except KeyboardInterrupt:
            print("Animation stopped")


@app.command()
def main(
    file: str = typer.Argument(..., help="Path to CSV file with trajectory waypoints"),
    loop: bool = typer.Option(
        False, "--loop", help="Loop the trajectory animation continuously"
    ),
    speed: float = typer.Option(
        1.0, "--speed", help="Animation speed factor (default: 1.0)"
    ),
):
    """
    Visualize a 3D trajectory with BlueROV2 model
    """
    # Convert to Path object
    file_path = Path(file)

    # Verify file exists
    if not file_path.exists():
        typer.echo(f"Error: File {file_path} not found")
        raise typer.Exit(code=1)

    # Load trajectory from CSV
    trajectory = load_trajectory_from_csv(file_path)
    if len(trajectory) == 0:
        typer.echo("Error: No valid waypoints found in the CSV file")
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {len(trajectory)} waypoints from {file_path}")

    # Get BlueROV2 model path
    try:
        import bluerov2_gym

        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            model_path = str(asset_path)
    except (ImportError, FileNotFoundError):
        typer.echo(
            "Warning: Could not find BlueROV2 model, using a simple cube instead"
        )
        model_path = None

    # Initialize visualizer and animate trajectory
    visualizer = TrajectoryVisualizer()

    if model_path:
        visualizer.setup_scene(model_path)
    else:
        # Use a simple cube if the model is not found
        visualizer.vis["vessel"].set_object(
            g.Box([0.5, 0.5, 0.25]), g.MeshLambertMaterial(color=0x0000FF)
        )

    visualizer.animate_trajectory(trajectory, loop=loop, speed=speed)


if __name__ == "__main__":
    app()
