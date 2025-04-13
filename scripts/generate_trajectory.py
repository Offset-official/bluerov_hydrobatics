#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import argparse
from datetime import datetime

def generate_spiral_trajectory(num_points=100, radius=5.0, height=10.0, num_loops=3):
    """
    Generate a 3D spiral trajectory
    
    Args:
        num_points: Number of waypoints to generate
        radius: Maximum radius of the spiral
        height: Total height of the spiral
        num_loops: Number of complete loops in the spiral
    
    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    t = np.linspace(0, num_loops * 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, height, num_points)
    
    return np.column_stack((x, y, z))

def generate_lemniscate_trajectory(num_points=100, scale=5.0, height_range=(-2.0, 2.0)):
    """
    Generate a 3D lemniscate (figure-eight) trajectory
    
    Args:
        num_points: Number of waypoints to generate
        scale: Scale factor for x and y coordinates
        height_range: Tuple of (min_height, max_height) for z oscillation
    
    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    # Add some variation in height
    z = np.linspace(height_range[0], height_range[1], num_points)
    
    return np.column_stack((x, y, z))

def generate_square_trajectory(num_points=100, side_length=5.0, height_range=(-1.0, 1.0)):
    """
    Generate a 3D square trajectory
    
    Args:
        num_points: Number of waypoints to generate
        side_length: Length of each side of the square
        height_range: Tuple of (min_height, max_height) for z oscillation
    
    Returns:
        numpy array of shape (num_points, 3) containing x, y, z coordinates
    """
    points_per_side = num_points // 4
    
    # Create four line segments for the square sides
    half_len = side_length / 2
    
    # First segment: moving along x-axis
    x1 = np.linspace(-half_len, half_len, points_per_side)
    y1 = np.ones(points_per_side) * -half_len
    
    # Second segment: moving along y-axis
    x2 = np.ones(points_per_side) * half_len
    y2 = np.linspace(-half_len, half_len, points_per_side)
    
    # Third segment: moving back along x-axis
    x3 = np.linspace(half_len, -half_len, points_per_side)
    y3 = np.ones(points_per_side) * half_len
    
    # Fourth segment: moving back along y-axis
    x4 = np.ones(points_per_side) * -half_len
    y4 = np.linspace(half_len, -half_len, points_per_side)
    
    # Combine segments
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    
    # Make sure the array has exactly num_points
    if len(x) < num_points:
        pad_length = num_points - len(x)
        x = np.pad(x, (0, pad_length), mode='edge')
        y = np.pad(y, (0, pad_length), mode='edge')
    elif len(x) > num_points:
        x = x[:num_points]
        y = y[:num_points]
    
    # Add some variation in height with a sinusoidal pattern
    z = np.linspace(height_range[0], height_range[1], num_points)
    z = height_range[0] + (height_range[1] - height_range[0]) * (np.sin(np.linspace(0, 2*np.pi, num_points)) + 1) / 2
    
    return np.column_stack((x, y, z))

def save_trajectory_to_csv(trajectory, filepath):
    """
    Save trajectory waypoints to a CSV file
    
    Args:
        trajectory: numpy array of shape (num_points, 3) containing x, y, z coordinates
        filepath: path to save the CSV file
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])  # Write header
        writer.writerows(trajectory)
    
    print(f"Trajectory saved to {filepath}")

def plot_trajectory(trajectory, title="3D Trajectory"):
    """
    Plot a 3D trajectory
    
    Args:
        trajectory: numpy array of shape (num_points, 3) containing x, y, z coordinates
        title: plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
    ax.plot(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 'go', markersize=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 'ro', markersize=10, label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate 3D trajectory waypoints and save to CSV')
    parser.add_argument('--type', type=str, default='spiral', choices=['spiral', 'lemniscate', 'square'],
                        help='Type of trajectory to generate (default: spiral)')
    parser.add_argument('--points', type=int, default=100, 
                        help='Number of waypoints to generate (default: 100)')
    parser.add_argument('--output', type=str, default='',
                        help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the trajectory after generating it')
    
    args = parser.parse_args()
    
    # Generate trajectory based on type
    if args.type == 'spiral':
        trajectory = generate_spiral_trajectory(num_points=args.points)
        traj_type = 'spiral'
    elif args.type == 'lemniscate':
        trajectory = generate_lemniscate_trajectory(num_points=args.points)
        traj_type = 'lemniscate'
    elif args.type == 'square':
        trajectory = generate_square_trajectory(num_points=args.points)
        traj_type = 'square'
    else:
        raise ValueError(f"Unknown trajectory type: {args.type}")
    
    # Generate output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../trajectories')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{traj_type}_trajectory_{timestamp}.csv")
    else:
        output_file = args.output
    
    # Save trajectory to CSV
    save_trajectory_to_csv(trajectory, output_file)
    
    # Plot if requested
    if args.plot:
        plot_trajectory(trajectory, title=f"{args.type.capitalize()} Trajectory ({args.points} points)")

if __name__ == "__main__":
    main()