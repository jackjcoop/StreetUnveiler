#!/usr/bin/env python3
"""
Visualization tool for custom VGGT point clouds in StreetUnveiler.

This script provides various visualization capabilities for the custom point clouds
and camera poses loaded from VGGT data.

Usage:
    python visualize_custom_pointcloud.py --custom_pc_path ./new_data/WOD_points --images ./new_data/WOD_undistorted_images --output_dir ./viz_output

Author: Generated for StreetUnveiler integration
"""

import sys
import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.custom_pointcloud_loader import CustomPointCloudLoader
from utils.general_utils import safe_state


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def visualize_point_cloud_3d(point_cloud, output_path: str, max_points: int = 10000, title: str = "Point Cloud"):
    """Create a 3D visualization of the point cloud."""
    logger = logging.getLogger(__name__)
    
    points = point_cloud.points.cpu().numpy()
    colors = point_cloud.colors.cpu().numpy()
    
    # Subsample for visualization if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, s=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title} ({len(points)} points)')
    
    # Set equal aspect ratio
    max_range = np.array([points.max()-points.min() for points in [points[:, 0], points[:, 1], points[:, 2]]]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"3D point cloud visualization saved to {output_path}")


def visualize_camera_poses(cameras, point_cloud, output_path: str, title: str = "Camera Poses"):
    """Visualize camera poses with the point cloud."""
    logger = logging.getLogger(__name__)
    
    points = point_cloud.points.cpu().numpy()
    colors = point_cloud.colors.cpu().numpy()
    
    # Subsample points for better visualization
    if len(points) > 5000:
        indices = np.random.choice(len(points), 5000, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=colors, s=0.1, alpha=0.3, label='Point Cloud')
    
    # Extract camera positions and orientations
    camera_positions = []
    camera_directions = []
    
    for camera in cameras:
        # Camera position is -R^T * T
        R = camera.R
        T = camera.T
        pos = -R.T @ T
        camera_positions.append(pos)
        
        # Camera forward direction (negative Z in camera frame)
        forward = R.T @ np.array([0, 0, -1])
        camera_directions.append(forward)
    
    camera_positions = np.array(camera_positions)
    camera_directions = np.array(camera_directions)
    
    # Plot camera positions
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
              c='red', s=50, marker='^', label='Cameras', alpha=0.8)
    
    # Plot camera directions as arrows (sample every 10th camera to avoid clutter)
    step = max(1, len(camera_positions) // 20)
    for i in range(0, len(camera_positions), step):
        pos = camera_positions[i]
        direction = camera_directions[i] * 2.0  # Scale for visibility
        ax.quiver(pos[0], pos[1], pos[2], 
                 direction[0], direction[1], direction[2],
                 color='blue', alpha=0.7, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title} ({len(cameras)} cameras)')
    ax.legend()
    
    # Set equal aspect ratio
    all_points = np.vstack([points, camera_positions])
    max_range = np.array([all_points.max()-all_points.min() for all_points in [all_points[:, 0], all_points[:, 1], all_points[:, 2]]]).max() / 2.0
    mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Camera poses visualization saved to {output_path}")


def visualize_data_statistics(loader: CustomPointCloudLoader, output_path: str):
    """Create visualizations of data statistics."""
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Load some segment data for analysis
    segment_stats = []
    for segment_dir in loader.segment_dirs[:min(5, len(loader.segment_dirs))]:  # Limit to 5 segments
        data = loader._load_numpy_data(segment_dir)
        points3d = data['points3d_unproj']
        
        # Calculate statistics
        depths = np.linalg.norm(points3d.reshape(-1, 3), axis=1)
        valid_depths = depths[depths > 0.1]  # Filter out invalid depths
        
        segment_stats.append({
            'name': segment_dir.name,
            'total_points': points3d.size // 3,
            'valid_points': len(valid_depths),
            'depth_mean': np.mean(valid_depths),
            'depth_std': np.std(valid_depths),
            'depth_min': np.min(valid_depths),
            'depth_max': np.max(valid_depths),
            'depths': valid_depths
        })
    
    # Plot 1: Depth distribution
    ax1 = axes[0, 0]
    all_depths = np.concatenate([stats['depths'] for stats in segment_stats])
    ax1.hist(all_depths, bins=100, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Depth (m)')
    ax1.set_ylabel('Count')
    ax1.set_title('Depth Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Points per segment
    ax2 = axes[0, 1]
    segment_names = [stats['name'][-20:] for stats in segment_stats]  # Last 20 chars
    valid_points = [stats['valid_points'] for stats in segment_stats]
    ax2.bar(range(len(segment_names)), valid_points)
    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Valid Points')
    ax2.set_title('Valid Points per Segment')
    ax2.set_xticks(range(len(segment_names)))
    ax2.set_xticklabels(segment_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Depth statistics per segment
    ax3 = axes[1, 0]
    depth_means = [stats['depth_mean'] for stats in segment_stats]
    depth_stds = [stats['depth_std'] for stats in segment_stats]
    x_pos = range(len(segment_names))
    ax3.errorbar(x_pos, depth_means, yerr=depth_stds, fmt='o-', capsize=5)
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Mean Depth per Segment')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(segment_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Point cloud size comparison
    ax4 = axes[1, 1]
    sizes_mb = [(stats['total_points'] * 3 * 4) / (1024 * 1024) for stats in segment_stats]  # Assuming float32
    ax4.pie(sizes_mb, labels=segment_names, autopct='%1.1f%%')
    ax4.set_title('Relative Point Cloud Sizes')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Data statistics visualization saved to {output_path}")


def create_projection_visualization(loader: CustomPointCloudLoader, output_path: str, camera_idx: int = 0):
    """Create visualization showing point projections onto camera images."""
    logger = logging.getLogger(__name__)
    
    if not loader.segment_dirs:
        logger.warning("No segments found for projection visualization")
        return
    
    # Load data from first segment
    segment_dir = loader.segment_dirs[0]
    data = loader._load_numpy_data(segment_dir)
    
    points3d = data['points3d_unproj'][camera_idx]  # Points for specific camera
    intrinsics = data['intrinsic'][camera_idx]
    
    # Load corresponding image if available
    images = loader._load_images_for_segment(segment_dir.name)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: 3D points
    ax1 = axes[0]
    ax1 = plt.subplot(121, projection='3d')
    
    # Sample points for visualization
    h, w = points3d.shape[:2]
    step = max(1, h // 50)  # Sample every step-th point
    sampled_points = points3d[::step, ::step].reshape(-1, 3)
    
    # Filter valid points
    depths = np.linalg.norm(sampled_points, axis=1)
    valid_mask = (depths > 0.1) & (depths < 100)
    valid_points = sampled_points[valid_mask]
    
    if len(valid_points) > 0:
        ax1.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'3D Points (Camera {camera_idx})')
    
    # Plot 2: Image with projected points (if available)
    ax2 = axes[1]
    if camera_idx < len(images) and images[camera_idx] is not None:
        # Convert tensor to numpy and handle different formats
        img = images[camera_idx]
        if hasattr(img, 'permute'):  # PyTorch tensor
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img
        
        ax2.imshow(img_np)
        ax2.set_title(f'Camera Image {camera_idx}')
        
        # Project some 3D points onto the image
        # This is a simplified projection - in practice, you'd use the full camera model
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Sample some points for projection visualization
        sample_indices = np.random.choice(len(valid_points), min(100, len(valid_points)), replace=False)
        sample_3d = valid_points[sample_indices]
        
        # Simple perspective projection
        projected_x = (sample_3d[:, 0] / sample_3d[:, 2]) * fx + cx
        projected_y = (sample_3d[:, 1] / sample_3d[:, 2]) * fy + cy
        
        # Filter points within image bounds
        img_h, img_w = img_np.shape[:2]
        in_bounds = (projected_x >= 0) & (projected_x < img_w) & (projected_y >= 0) & (projected_y < img_h)
        
        if np.any(in_bounds):
            ax2.scatter(projected_x[in_bounds], projected_y[in_bounds], c='red', s=10, alpha=0.7)
            ax2.set_title(f'Camera Image {camera_idx} with Projected Points')
    else:
        ax2.text(0.5, 0.5, 'No image available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Camera {camera_idx} (No Image)')
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Projection visualization saved to {output_path}")


def generate_summary_report(loader: CustomPointCloudLoader, point_cloud, cameras, output_path: str):
    """Generate a text summary report of the loaded data."""
    logger = logging.getLogger(__name__)
    
    # Collect statistics
    points = point_cloud.points.cpu().numpy()
    colors = point_cloud.colors.cpu().numpy()
    semantics = point_cloud.semantics.cpu().numpy()
    
    report = {
        "summary": {
            "segments": len(loader.segment_dirs),
            "total_points": len(points),
            "total_cameras": len(cameras)
        },
        "point_cloud": {
            "points_shape": list(points.shape),
            "colors_shape": list(colors.shape),
            "semantics_shape": list(semantics.shape),
            "points_range": {
                "min": [float(points.min(axis=0)[i]) for i in range(3)],
                "max": [float(points.max(axis=0)[i]) for i in range(3)],
                "mean": [float(points.mean(axis=0)[i]) for i in range(3)],
                "std": [float(points.std(axis=0)[i]) for i in range(3)]
            },
            "colors_range": {
                "min": [float(colors.min(axis=0)[i]) for i in range(3)],
                "max": [float(colors.max(axis=0)[i]) for i in range(3)],
                "mean": [float(colors.mean(axis=0)[i]) for i in range(3)]
            }
        },
        "cameras": {
            "count": len(cameras),
            "fov_range": {
                "fovx_min": float(min(cam.FoVx for cam in cameras)),
                "fovx_max": float(max(cam.FoVx for cam in cameras)),
                "fovy_min": float(min(cam.FoVy for cam in cameras)),
                "fovy_max": float(max(cam.FoVy for cam in cameras))
            },
            "image_sizes": list(set((cam.image_width, cam.image_height) for cam in cameras))
        },
        "segments": []
    }
    
    # Per-segment statistics
    for segment_dir in loader.segment_dirs:
        try:
            data = loader._load_numpy_data(segment_dir)
            points3d = data['points3d_unproj']
            intrinsics = data['intrinsic']
            extrinsics = data['extrinsic']
            
            segment_info = {
                "name": segment_dir.name,
                "frames": points3d.shape[0],
                "resolution": [int(points3d.shape[1]), int(points3d.shape[2])],
                "total_3d_points": int(points3d.size // 3),
                "intrinsics_shape": list(intrinsics.shape),
                "extrinsics_shape": list(extrinsics.shape),
                "available_data": list(data.keys())
            }
            
            report["segments"].append(segment_info)
            
        except Exception as e:
            logger.warning(f"Could not process segment {segment_dir.name}: {e}")
    
    # Save as JSON
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save as readable text
    with open(output_path, 'w') as f:
        f.write("CUSTOM POINT CLOUD DATA SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERVIEW:\n")
        f.write(f"  Total Segments: {report['summary']['segments']}\n")
        f.write(f"  Total Points: {report['summary']['total_points']:,}\n")
        f.write(f"  Total Cameras: {report['summary']['total_cameras']}\n\n")
        
        f.write("POINT CLOUD DETAILS:\n")
        pc = report['point_cloud']
        f.write(f"  Points Shape: {pc['points_shape']}\n")
        f.write(f"  Colors Shape: {pc['colors_shape']}\n")
        f.write(f"  Semantics Shape: {pc['semantics_shape']}\n")
        f.write(f"  Spatial Range: X[{pc['points_range']['min'][0]:.3f}, {pc['points_range']['max'][0]:.3f}], ")
        f.write(f"Y[{pc['points_range']['min'][1]:.3f}, {pc['points_range']['max'][1]:.3f}], ")
        f.write(f"Z[{pc['points_range']['min'][2]:.3f}, {pc['points_range']['max'][2]:.3f}]\n")
        f.write(f"  Color Range: R[{pc['colors_range']['min'][0]:.3f}, {pc['colors_range']['max'][0]:.3f}], ")
        f.write(f"G[{pc['colors_range']['min'][1]:.3f}, {pc['colors_range']['max'][1]:.3f}], ")
        f.write(f"B[{pc['colors_range']['min'][2]:.3f}, {pc['colors_range']['max'][2]:.3f}]\n\n")
        
        f.write("CAMERA DETAILS:\n")
        cam = report['cameras']
        f.write(f"  Camera Count: {cam['count']}\n")
        f.write(f"  FoV Range: X[{cam['fov_range']['fovx_min']:.4f}, {cam['fov_range']['fovx_max']:.4f}], ")
        f.write(f"Y[{cam['fov_range']['fovy_min']:.4f}, {cam['fov_range']['fovy_max']:.4f}]\n")
        f.write(f"  Image Sizes: {cam['image_sizes']}\n\n")
        
        f.write("SEGMENT DETAILS:\n")
        for seg in report['segments']:
            f.write(f"  {seg['name']}:\n")
            f.write(f"    Frames: {seg['frames']}\n")
            f.write(f"    Resolution: {seg['resolution']}\n")
            f.write(f"    3D Points: {seg['total_3d_points']:,}\n")
            f.write(f"    Available Data: {', '.join(seg['available_data'])}\n")
    
    logger.info(f"Summary report saved to {output_path} and {json_path}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize custom point cloud data")
    parser.add_argument("--custom_pc_path", type=str, required=True,
                       help="Path to custom point cloud data directory")
    parser.add_argument("--images", type=str, default=None,
                       help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="./visualization_output",
                       help="Directory to save visualizations")
    parser.add_argument("--max_points", type=int, default=10000,
                       help="Maximum points to visualize (for performance)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize system state
    safe_state(quiet=False)
    
    logger.info("Starting custom point cloud visualization...")
    logger.info(f"Custom PC path: {args.custom_pc_path}")
    logger.info(f"Images path: {args.images}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize loader
        loader = CustomPointCloudLoader(args.custom_pc_path, args.images)
        
        # Load data
        logger.info("Loading point cloud...")
        point_cloud = loader.load_point_cloud()
        
        logger.info("Loading cameras...")
        cameras = loader.load_cameras()
        
        # Generate visualizations
        logger.info("Creating visualizations...")
        
        # 3D Point Cloud
        visualize_point_cloud_3d(
            point_cloud, 
            str(output_dir / "point_cloud_3d.png"),
            max_points=args.max_points,
            title="Custom Point Cloud"
        )
        
        # Camera Poses
        if cameras:
            visualize_camera_poses(
                cameras,
                point_cloud,
                str(output_dir / "camera_poses.png"),
                title="Camera Poses with Point Cloud"
            )
        
        # Data Statistics
        visualize_data_statistics(
            loader,
            str(output_dir / "data_statistics.png")
        )
        
        # Projection Visualization
        create_projection_visualization(
            loader,
            str(output_dir / "projection_example.png"),
            camera_idx=0
        )
        
        # Summary Report
        generate_summary_report(
            loader,
            point_cloud,
            cameras,
            str(output_dir / "summary_report.txt")
        )
        
        logger.info(f"✓ All visualizations completed successfully!")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Generated files:")
        for file in output_dir.glob("*"):
            logger.info(f"    - {file.name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())