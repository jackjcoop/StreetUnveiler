"""
Custom Point Cloud Reader for StreetUnveiler

This module reads custom VGGT-derived point cloud data and creates
scene information compatible with StreetUnveiler's pipeline.

Author: Generated for StreetUnveiler integration
"""

import numpy as np
from pathlib import Path
from typing import NamedTuple, List
import os
import tempfile
import logging

from scene.custom_pointcloud_loader import CustomPointCloudLoader
from utils.pcd_utils import SemanticPointCloud
from scene.cameras import Camera
from utils.graphics_utils import focal2fov


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FoVy: float
    FoVx: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: SemanticPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str
    reference_ply_path: str


def readCustomSceneInfo(custom_pc_path, images_path, eval_split=False):
    """
    Read custom point cloud scene information.
    
    Args:
        custom_pc_path: Path to custom point cloud data
        images_path: Path to corresponding images
        eval_split: Whether to split into train/test sets
        
    Returns:
        SceneInfo object
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading custom scene from {custom_pc_path}")
    
    # Initialize custom point cloud loader
    loader = CustomPointCloudLoader(custom_pc_path, images_path)
    
    # Get scene info from loader
    scene_data = loader.get_scene_info(eval_split=eval_split)
    
    # Convert Camera objects to CameraInfo
    def camera_to_camera_info(camera: Camera) -> CameraInfo:
        return CameraInfo(
            uid=camera.uid,
            R=camera.R,
            T=camera.T,
            FoVy=camera.FoVy,
            FoVx=camera.FoVx,
            image=camera.original_image.permute(1, 2, 0).cpu().numpy() if camera.original_image is not None else None,
            image_path=camera.image_name,
            image_name=camera.image_name,
            width=camera.image_width,
            height=camera.image_height
        )
    
    train_cameras = [camera_to_camera_info(cam) for cam in scene_data['train_cameras']]
    test_cameras = [camera_to_camera_info(cam) for cam in scene_data['test_cameras']]
    
    # Create temporary PLY files for compatibility
    temp_dir = Path(tempfile.gettempdir()) / "streetunveiler_custom"
    temp_dir.mkdir(exist_ok=True)
    
    ply_path = str(temp_dir / "custom_point_cloud.ply")
    reference_ply_path = str(temp_dir / "reference_point_cloud.ply")
    
    # Save point cloud as PLY (will be handled by existing StreetUnveiler logic)
    point_cloud = scene_data['point_cloud']
    save_point_cloud_as_ply(point_cloud, ply_path)
    save_point_cloud_as_ply(point_cloud, reference_ply_path)
    
    logger.info(f"Created {len(train_cameras)} train cameras and {len(test_cameras)} test cameras")
    logger.info(f"Point cloud contains {len(point_cloud.points)} points")
    
    return SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=scene_data['nerf_normalization'],
        ply_path=ply_path,
        reference_ply_path=reference_ply_path
    )


def save_point_cloud_as_ply(point_cloud: SemanticPointCloud, output_path: str):
    """Save a SemanticPointCloud as a PLY file."""
    import struct
    
    points = point_cloud.points.cpu().numpy()
    colors = (point_cloud.colors.cpu().numpy() * 255).astype(np.uint8)
    semantics = point_cloud.semantics.cpu().numpy()
    
    # PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int semantic
end_header
"""
    
    with open(output_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        
        for i in range(len(points)):
            # Write x, y, z as floats
            f.write(struct.pack('<fff', points[i, 0], points[i, 1], points[i, 2]))
            # Write r, g, b as unsigned chars
            f.write(struct.pack('<BBB', colors[i, 0], colors[i, 1], colors[i, 2]))
            # Write semantic as int
            f.write(struct.pack('<i', int(semantics[i])))


def readCustomSceneInfoAugmented(custom_pc_path, images_path, original_point_cloud: SemanticPointCloud, eval_split=False):
    """
    Read custom point cloud scene information and combine with existing point cloud.
    
    Args:
        custom_pc_path: Path to custom point cloud data
        images_path: Path to corresponding images
        original_point_cloud: Existing point cloud to augment
        eval_split: Whether to split into train/test sets
        
    Returns:
        SceneInfo object with augmented point cloud
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading custom scene for augmentation from {custom_pc_path}")
    
    # Load custom scene data
    custom_scene = readCustomSceneInfo(custom_pc_path, images_path, eval_split)
    
    # Combine point clouds
    import torch
    
    combined_points = torch.cat([original_point_cloud.points, custom_scene.point_cloud.points], dim=0)
    combined_colors = torch.cat([original_point_cloud.colors, custom_scene.point_cloud.colors], dim=0)
    combined_semantics = torch.cat([original_point_cloud.semantics, custom_scene.point_cloud.semantics], dim=0)
    
    # Create augmented point cloud
    augmented_point_cloud = SemanticPointCloud(
        points=combined_points,
        colors=combined_colors,
        semantics=combined_semantics
    )
    
    # Create temporary PLY files for augmented point cloud
    temp_dir = Path(tempfile.gettempdir()) / "streetunveiler_custom"
    temp_dir.mkdir(exist_ok=True)
    
    ply_path = str(temp_dir / "augmented_point_cloud.ply")
    reference_ply_path = str(temp_dir / "reference_augmented_point_cloud.ply")
    
    save_point_cloud_as_ply(augmented_point_cloud, ply_path)
    save_point_cloud_as_ply(augmented_point_cloud, reference_ply_path)
    
    logger.info(f"Augmented point cloud contains {len(augmented_point_cloud.points)} points "
               f"(original: {len(original_point_cloud.points)}, custom: {len(custom_scene.point_cloud.points)})")
    
    # Return SceneInfo with augmented point cloud but original cameras
    # (assuming we want to use original cameras, not custom ones for augmented mode)
    return SceneInfo(
        point_cloud=augmented_point_cloud,
        train_cameras=custom_scene.train_cameras,  # Could be modified based on requirements
        test_cameras=custom_scene.test_cameras,
        nerf_normalization=custom_scene.nerf_normalization,
        ply_path=ply_path,
        reference_ply_path=reference_ply_path
    )