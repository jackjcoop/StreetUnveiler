"""
Custom VGGT Point Cloud Loader for StreetUnveiler

This module handles loading and processing of custom VGGT-derived point clouds
as an alternative or supplement to COLMAP/LiDAR initialization.

Author: Generated for StreetUnveiler integration
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import os
from utils.pcd_utils import SemanticPointCloud
from utils.general_utils import PILtoTorch
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import logging

class CustomPointCloudLoader:
    """Loader for VGGT-derived custom point cloud data."""
    
    def __init__(self, custom_pc_path: str, images_path: str = None):
        """
        Initialize the custom point cloud loader.
        
        Args:
            custom_pc_path: Path to the custom point cloud data directory
            images_path: Path to the corresponding images directory
        """
        self.custom_pc_path = Path(custom_pc_path)
        self.images_path = Path(images_path) if images_path else None
        self.segment_dirs = []
        
        # Find all segment directories
        self._find_segment_directories()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _find_segment_directories(self):
        """Find all segment directories containing point cloud data."""
        if not self.custom_pc_path.exists():
            raise FileNotFoundError(f"Custom point cloud path does not exist: {self.custom_pc_path}")
            
        # Look for segment directories
        for item in self.custom_pc_path.iterdir():
            if item.is_dir() and item.name.startswith('segment-'):
                # Verify required files exist
                required_files = [
                    'points3d_unproj.npy',
                    'intrinsic.npy', 
                    'extrinsic.npy'
                ]
                
                if all((item / f).exists() for f in required_files):
                    self.segment_dirs.append(item)
                    
        self.logger.info(f"Found {len(self.segment_dirs)} valid segment directories")
        
    def _load_numpy_data(self, segment_dir: Path) -> Dict[str, np.ndarray]:
        """Load all numpy data from a segment directory."""
        data = {}
        
        try:
            # Load required files
            data['points3d_unproj'] = np.load(segment_dir / 'points3d_unproj.npy')
            data['intrinsic'] = np.load(segment_dir / 'intrinsic.npy') 
            data['extrinsic'] = np.load(segment_dir / 'extrinsic.npy')
            
            # Load optional files
            for optional_file in ['depth_map.npy', 'point_map.npy', 'depth_conf.npy', 'point_conf.npy']:
                file_path = segment_dir / optional_file
                if file_path.exists():
                    data[optional_file.replace('.npy', '')] = np.load(file_path)
                    
            self.logger.info(f"Loaded data from {segment_dir.name}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {segment_dir}: {e}")
            raise
            
    def _extract_point_cloud_from_segment(self, data: Dict[str, np.ndarray], 
                                        confidence_threshold: float = 0.5,
                                        max_depth: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 3D point cloud from segment data.
        
        Args:
            data: Dictionary containing loaded numpy arrays
            confidence_threshold: Minimum confidence for point inclusion
            max_depth: Maximum depth for point inclusion
            
        Returns:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors (0-1 range)
        """
        points3d = data['points3d_unproj']  # Shape: (frames, height, width, 3)
        intrinsics = data['intrinsic']      # Shape: (frames, 3, 3)
        extrinsics = data['extrinsic']      # Shape: (frames, 3, 4)
        
        n_frames, height, width = points3d.shape[:3]
        
        # Initialize lists to collect points and colors
        all_points = []
        all_colors = []
        
        # Process each frame
        for frame_idx in range(n_frames):
            frame_points = points3d[frame_idx]  # (height, width, 3)
            
            # Filter points based on depth
            depths = np.linalg.norm(frame_points, axis=2)
            valid_mask = (depths > 0.1) & (depths < max_depth)
            
            # Apply confidence filtering if available
            if 'point_conf' in data:
                conf = data['point_conf'][frame_idx]
                if conf.ndim == 3:
                    conf = conf.squeeze(-1)
                valid_mask &= (conf > confidence_threshold)
            
            # Extract valid points
            valid_points = frame_points[valid_mask]
            
            if len(valid_points) > 0:
                all_points.append(valid_points)
                
                # Generate colors (placeholder - could be improved with actual color data)
                # For now, use a simple depth-based coloring
                frame_depths = depths[valid_mask]
                normalized_depths = np.clip(frame_depths / max_depth, 0, 1)
                colors = np.zeros((len(valid_points), 3))
                colors[:, 0] = 1.0 - normalized_depths  # Red decreases with depth
                colors[:, 1] = normalized_depths        # Green increases with depth  
                colors[:, 2] = 0.5                      # Blue constant
                
                all_colors.append(colors)
                
        if all_points:
            points = np.vstack(all_points)
            colors = np.vstack(all_colors)
        else:
            self.logger.warning("No valid points found in segment")
            points = np.empty((0, 3))
            colors = np.empty((0, 3))
            
        return points, colors
        
    def _load_images_for_segment(self, segment_name: str) -> List[np.ndarray]:
        """Load images corresponding to a segment."""
        if not self.images_path:
            return []
            
        segment_image_dir = self.images_path / segment_name
        if not segment_image_dir.exists():
            self.logger.warning(f"No images found for segment {segment_name}")
            return []
            
        images = []
        image_files = sorted(segment_image_dir.glob('*.jpg'))
        
        for img_file in image_files:
            try:
                # Load using PIL and convert to tensor format expected by StreetUnveiler
                img_tensor = PILtoTorch(img_file, (None, None))
                images.append(img_tensor)
            except Exception as e:
                self.logger.warning(f"Failed to load image {img_file}: {e}")
                
        return images
        
    def _create_camera_objects(self, data: Dict[str, np.ndarray], 
                              images: List[np.ndarray], 
                              segment_name: str) -> List[Camera]:
        """Create Camera objects from the loaded data."""
        intrinsics = data['intrinsic']      # Shape: (frames, 3, 3)
        extrinsics = data['extrinsic']      # Shape: (frames, 3, 4)
        
        n_frames = intrinsics.shape[0]
        cameras = []
        
        for frame_idx in range(n_frames):
            # Extract intrinsic parameters
            K = intrinsics[frame_idx]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Extract extrinsic parameters (3x4 matrix)
            extr_3x4 = extrinsics[frame_idx]
            
            # Convert to 4x4 homogeneous matrix
            extr_4x4 = np.eye(4)
            extr_4x4[:3, :] = extr_3x4
            
            # Extract rotation and translation
            R = extr_4x4[:3, :3]
            T = extr_4x4[:3, 3]
            
            # Image dimensions (from the data shape)
            height, width = 518, 518  # From the data analysis
            
            # Calculate FoV from focal lengths
            FoVx = focal2fov(fx, width)
            FoVy = focal2fov(fy, height)
            
            # Get corresponding image if available
            image = images[frame_idx] if frame_idx < len(images) else None
            
            # Create Camera object
            camera = Camera(
                colmap_id=frame_idx,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=image,
                gt_alpha_mask=None,
                image_name=f"{segment_name}_{frame_idx:05d}",
                uid=f"{segment_name}_{frame_idx}",
                data_device="cuda"
            )
            
            cameras.append(camera)
            
        return cameras
        
    def load_point_cloud(self, mode: str = "replace", 
                        confidence_threshold: float = 0.5,
                        max_points_per_segment: int = 50000) -> SemanticPointCloud:
        """
        Load and combine point clouds from all segments.
        
        Args:
            mode: Loading mode ("replace" or "augment")
            confidence_threshold: Minimum confidence for point inclusion
            max_points_per_segment: Maximum points to sample per segment
            
        Returns:
            SemanticPointCloud object
        """
        if not self.segment_dirs:
            raise ValueError("No valid segment directories found")
            
        all_points = []
        all_colors = []
        all_semantics = []
        
        for segment_dir in self.segment_dirs:
            self.logger.info(f"Processing segment: {segment_dir.name}")
            
            # Load segment data
            data = self._load_numpy_data(segment_dir)
            
            # Extract point cloud
            points, colors = self._extract_point_cloud_from_segment(
                data, confidence_threshold=confidence_threshold
            )
            
            if len(points) > 0:
                # Downsample if necessary
                if len(points) > max_points_per_segment:
                    indices = np.random.choice(len(points), max_points_per_segment, replace=False)
                    points = points[indices]
                    colors = colors[indices]
                
                all_points.append(points)
                all_colors.append(colors)
                
                # Assign default semantic labels (could be improved with actual semantic data)
                semantics = np.zeros(len(points), dtype=np.int32)  # Default: background
                all_semantics.append(semantics)
                
        if all_points:
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            combined_semantics = np.hstack(all_semantics)
        else:
            raise ValueError("No valid points found in any segment")
            
        self.logger.info(f"Loaded {len(combined_points)} total points from {len(self.segment_dirs)} segments")
        
        # Create SemanticPointCloud object
        pcd = SemanticPointCloud(
            points=torch.from_numpy(combined_points.astype(np.float32)),
            colors=torch.from_numpy(combined_colors.astype(np.float32)), 
            semantics=torch.from_numpy(combined_semantics.astype(np.int32))
        )
        
        return pcd
        
    def load_cameras(self) -> List[Camera]:
        """Load camera objects from all segments."""
        all_cameras = []
        
        for segment_dir in self.segment_dirs:
            segment_name = segment_dir.name
            
            # Load segment data
            data = self._load_numpy_data(segment_dir)
            
            # Load corresponding images
            images = self._load_images_for_segment(segment_name)
            
            # Create camera objects
            cameras = self._create_camera_objects(data, images, segment_name)
            all_cameras.extend(cameras)
            
        self.logger.info(f"Created {len(all_cameras)} camera objects")
        return all_cameras
        
    def get_scene_info(self, eval_split: bool = False) -> Dict:
        """
        Create scene info dictionary compatible with StreetUnveiler.
        
        Args:
            eval_split: Whether to split into train/test sets
            
        Returns:
            Dictionary containing scene information
        """
        # Load point cloud
        point_cloud = self.load_point_cloud()
        
        # Load cameras
        cameras = self.load_cameras()
        
        # Split cameras into train/test if requested
        if eval_split and len(cameras) > 10:
            split_idx = int(len(cameras) * 0.8)
            train_cameras = cameras[:split_idx]
            test_cameras = cameras[split_idx:]
        else:
            train_cameras = cameras
            test_cameras = []
            
        # Calculate scene bounds for normalization
        points_np = point_cloud.points.numpy()
        scene_center = np.mean(points_np, axis=0)
        scene_radius = np.max(np.linalg.norm(points_np - scene_center, axis=1))
        
        scene_info = {
            'point_cloud': point_cloud,
            'train_cameras': train_cameras,
            'test_cameras': test_cameras,
            'nerf_normalization': {
                'radius': float(scene_radius),
                'translate': scene_center.tolist()
            },
            'ply_path': None,  # Will be set by caller
            'reference_ply_path': None  # Will be set by caller
        }
        
        return scene_info


def load_custom_point_cloud(custom_pc_path: str, 
                           images_path: str = None,
                           confidence_threshold: float = 0.5,
                           max_points_per_segment: int = 50000) -> SemanticPointCloud:
    """
    Convenience function to load a custom point cloud.
    
    Args:
        custom_pc_path: Path to custom point cloud data
        images_path: Path to corresponding images
        confidence_threshold: Minimum confidence for point inclusion
        max_points_per_segment: Maximum points per segment
        
    Returns:
        SemanticPointCloud object
    """
    loader = CustomPointCloudLoader(custom_pc_path, images_path)
    return loader.load_point_cloud(
        confidence_threshold=confidence_threshold,
        max_points_per_segment=max_points_per_segment
    )