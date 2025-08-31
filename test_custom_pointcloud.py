#!/usr/bin/env python3
"""
Test script for custom VGGT point cloud integration with StreetUnveiler.

This script validates the custom point cloud loading and initialization
functionality before running the full training pipeline.

Usage:
    python test_custom_pointcloud.py --custom_pc_path ./new_data/WOD_points --images ./new_data/WOD_undistorted_images

Author: Generated for StreetUnveiler integration
"""

import sys
import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.custom_pointcloud_loader import CustomPointCloudLoader, load_custom_point_cloud
from scene.dataset_readers.custom import readCustomSceneInfo
from utils.general_utils import safe_state


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_custom_pointcloud.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def test_custom_loader(custom_pc_path: str, images_path: str = None):
    """Test the custom point cloud loader."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Testing Custom Point Cloud Loader")
    logger.info("=" * 60)
    
    try:
        # Initialize loader
        loader = CustomPointCloudLoader(custom_pc_path, images_path)
        logger.info(f"✓ CustomPointCloudLoader initialized successfully")
        logger.info(f"  Found {len(loader.segment_dirs)} segment directories")
        
        # Test point cloud loading
        logger.info("\nTesting point cloud loading...")
        point_cloud = loader.load_point_cloud()
        logger.info(f"✓ Point cloud loaded successfully")
        logger.info(f"  Points shape: {point_cloud.points.shape}")
        logger.info(f"  Colors shape: {point_cloud.colors.shape}")
        logger.info(f"  Semantics shape: {point_cloud.semantics.shape}")
        logger.info(f"  Points range: [{point_cloud.points.min():.3f}, {point_cloud.points.max():.3f}]")
        logger.info(f"  Colors range: [{point_cloud.colors.min():.3f}, {point_cloud.colors.max():.3f}]")
        
        # Test camera loading
        if images_path:
            logger.info("\nTesting camera loading...")
            cameras = loader.load_cameras()
            logger.info(f"✓ Cameras loaded successfully")
            logger.info(f"  Number of cameras: {len(cameras)}")
            if cameras:
                cam = cameras[0]
                logger.info(f"  First camera:")
                logger.info(f"    UID: {cam.uid}")
                logger.info(f"    FoVx: {cam.FoVx:.4f}, FoVy: {cam.FoVy:.4f}")
                logger.info(f"    Image size: {cam.image_width}x{cam.image_height}")
                
        # Test scene info creation
        logger.info("\nTesting scene info creation...")
        scene_info = loader.get_scene_info(eval_split=True)
        logger.info(f"✓ Scene info created successfully")
        logger.info(f"  Train cameras: {len(scene_info['train_cameras'])}")
        logger.info(f"  Test cameras: {len(scene_info['test_cameras'])}")
        logger.info(f"  Scene radius: {scene_info['nerf_normalization']['radius']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CustomPointCloudLoader test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_dataset_reader(custom_pc_path: str, images_path: str = None):
    """Test the custom dataset reader."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("Testing Custom Dataset Reader")
    logger.info("=" * 60)
    
    try:
        # Test scene info reading
        scene_info = readCustomSceneInfo(custom_pc_path, images_path, eval_split=True)
        logger.info(f"✓ Scene info read successfully")
        logger.info(f"  Point cloud points: {len(scene_info.point_cloud.points)}")
        logger.info(f"  Train cameras: {len(scene_info.train_cameras)}")
        logger.info(f"  Test cameras: {len(scene_info.test_cameras)}")
        logger.info(f"  PLY path: {scene_info.ply_path}")
        logger.info(f"  Reference PLY path: {scene_info.reference_ply_path}")
        
        # Verify PLY files were created
        if os.path.exists(scene_info.ply_path):
            logger.info(f"✓ PLY file created: {scene_info.ply_path}")
            file_size = os.path.getsize(scene_info.ply_path) / (1024 * 1024)  # MB
            logger.info(f"  File size: {file_size:.2f} MB")
        else:
            logger.warning(f"✗ PLY file not found: {scene_info.ply_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset reader test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_integration_with_scene(custom_pc_path: str, images_path: str = None):
    """Test integration with StreetUnveiler Scene class."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("Testing Scene Integration")
    logger.info("=" * 60)
    
    try:
        from scene import Scene
        from scene.gaussian_model import GaussianModel
        from arguments import ModelParams
        from argparse import Namespace
        
        # Create mock arguments
        args = Namespace()
        args.sh_degree = 3
        args.source_path = ""  # Not used for custom
        args.colmap_path = ""
        args.model_path = "./output/test_custom"
        args.start_frame = None
        args.end_frame = None
        args.images = images_path if images_path else "images"
        args.resolution = -1
        args.white_background = False
        args.data_device = "cuda"
        args.eval = True
        
        # Custom point cloud arguments
        args.custom_pc_path = custom_pc_path
        args.use_custom_init = True
        args.custom_init_mode = "replace"
        args.custom_confidence_threshold = 0.5
        args.custom_max_points_per_segment = 50000
        
        # Create model params
        model_params = ModelParams(None, sentinel=True)
        
        # Extract args (this processes the custom_pc_path)
        dataset = model_params.extract(args)
        
        # Create Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Create scene with custom initialization
        os.makedirs(dataset.model_path, exist_ok=True)
        scene = Scene(dataset, gaussians)
        
        logger.info(f"✓ Scene created successfully with custom point cloud")
        logger.info(f"  Scene type: {scene.scene_type}")
        logger.info(f"  Train cameras: {len(scene.getTrainCameras())}")
        logger.info(f"  Test cameras: {len(scene.getTestCameras())}")
        logger.info(f"  Gaussians points: {gaussians.get_xyz.shape[0]}")
        logger.info(f"  Camera extent: {scene.cameras_extent:.3f}")
        
        # Test some basic Scene operations
        if len(scene.getTrainCameras()) > 0:
            cam = scene.getTrainCameras()[0]
            logger.info(f"  First camera FoV: {cam.FoVx:.4f} x {cam.FoVy:.4f}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Scene integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_convenience_function(custom_pc_path: str):
    """Test the convenience function."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("Testing Convenience Function")
    logger.info("=" * 60)
    
    try:
        point_cloud = load_custom_point_cloud(
            custom_pc_path=custom_pc_path,
            confidence_threshold=0.3,
            max_points_per_segment=10000
        )
        
        logger.info(f"✓ Convenience function worked successfully")
        logger.info(f"  Points shape: {point_cloud.points.shape}")
        logger.info(f"  Device: {point_cloud.points.device}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Convenience function test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test custom point cloud integration")
    parser.add_argument("--custom_pc_path", type=str, required=True,
                       help="Path to custom point cloud data directory")
    parser.add_argument("--images", type=str, default=None,
                       help="Path to images directory")
    parser.add_argument("--skip_scene_test", action="store_true",
                       help="Skip scene integration test (requires CUDA)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize system state
    safe_state(quiet=False)
    
    logger.info("Starting custom point cloud integration tests...")
    logger.info(f"Custom PC path: {args.custom_pc_path}")
    logger.info(f"Images path: {args.images}")
    
    # Verify paths exist
    if not os.path.exists(args.custom_pc_path):
        logger.error(f"Custom point cloud path does not exist: {args.custom_pc_path}")
        return 1
        
    if args.images and not os.path.exists(args.images):
        logger.error(f"Images path does not exist: {args.images}")
        return 1
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if test_custom_loader(args.custom_pc_path, args.images):
        tests_passed += 1
        
    if test_dataset_reader(args.custom_pc_path, args.images):
        tests_passed += 1
        
    if test_convenience_function(args.custom_pc_path):
        tests_passed += 1
        
    if not args.skip_scene_test:
        if test_integration_with_scene(args.custom_pc_path, args.images):
            tests_passed += 1
    else:
        total_tests -= 1
        logger.info("Skipped scene integration test")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! Custom point cloud integration is working.")
        return 0
    else:
        logger.error(f"❌ {total_tests - tests_passed} test(s) failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())