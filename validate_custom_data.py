#!/usr/bin/env python3
"""
Basic validation script for custom VGGT point cloud data.
This script validates data without requiring PyTorch dependencies.

Usage:
    python validate_custom_data.py --custom_pc_path ./new_data/WOD_points --images ./new_data/WOD_undistorted_images

Author: Generated for StreetUnveiler integration
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_segment_data(segment_dir: Path, logger):
    """Validate data files in a segment directory."""
    logger.info(f"Validating segment: {segment_dir.name}")
    
    validation_results = {
        'segment_name': segment_dir.name,
        'files_found': [],
        'files_missing': [],
        'data_shapes': {},
        'data_ranges': {},
        'validation_passed': True,
        'errors': []
    }
    
    # Required files
    required_files = [
        'points3d_unproj.npy',
        'intrinsic.npy', 
        'extrinsic.npy'
    ]
    
    # Optional files
    optional_files = [
        'depth_map.npy',
        'point_map.npy', 
        'depth_conf.npy',
        'point_conf.npy'
    ]
    
    try:
        # Check required files
        for file_name in required_files:
            file_path = segment_dir / file_name
            if file_path.exists():
                validation_results['files_found'].append(file_name)
                
                # Load and validate the data
                try:
                    data = np.load(file_path)
                    validation_results['data_shapes'][file_name] = list(data.shape)
                    validation_results['data_ranges'][file_name] = {
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'dtype': str(data.dtype)
                    }
                    logger.info(f"  ✓ {file_name}: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")
                except Exception as e:
                    validation_results['errors'].append(f"Failed to load {file_name}: {e}")
                    validation_results['validation_passed'] = False
                    logger.error(f"  ✗ {file_name}: Failed to load - {e}")
            else:
                validation_results['files_missing'].append(file_name)
                validation_results['validation_passed'] = False
                logger.error(f"  ✗ Required file missing: {file_name}")
        
        # Check optional files
        for file_name in optional_files:
            file_path = segment_dir / file_name
            if file_path.exists():
                validation_results['files_found'].append(file_name)
                try:
                    data = np.load(file_path)
                    validation_results['data_shapes'][file_name] = list(data.shape)
                    validation_results['data_ranges'][file_name] = {
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'dtype': str(data.dtype)
                    }
                    logger.info(f"  ○ {file_name}: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")
                except Exception as e:
                    validation_results['errors'].append(f"Failed to load optional {file_name}: {e}")
                    logger.warning(f"  ⚠ {file_name}: Failed to load - {e}")
        
        # Validate data consistency if all required files are present
        if len(validation_results['files_missing']) == 0:
            try:
                points3d = np.load(segment_dir / 'points3d_unproj.npy')
                intrinsic = np.load(segment_dir / 'intrinsic.npy')
                extrinsic = np.load(segment_dir / 'extrinsic.npy')
                
                # Check shape consistency
                n_frames_points = points3d.shape[0]
                n_frames_intrinsic = intrinsic.shape[0]
                n_frames_extrinsic = extrinsic.shape[0]
                
                if n_frames_points == n_frames_intrinsic == n_frames_extrinsic:
                    logger.info(f"  ✓ Frame consistency: {n_frames_points} frames")
                    validation_results['frame_count'] = n_frames_points
                else:
                    error_msg = f"Frame count mismatch: points3d={n_frames_points}, intrinsic={n_frames_intrinsic}, extrinsic={n_frames_extrinsic}"
                    validation_results['errors'].append(error_msg)
                    validation_results['validation_passed'] = False
                    logger.error(f"  ✗ {error_msg}")
                
                # Check intrinsic matrix format (should be 3x3)
                if intrinsic.shape[1:] == (3, 3):
                    logger.info(f"  ✓ Intrinsic matrices have correct shape (3x3)")
                    
                    # Check if intrinsic matrices look reasonable
                    sample_K = intrinsic[0]
                    fx, fy = sample_K[0, 0], sample_K[1, 1]
                    cx, cy = sample_K[0, 2], sample_K[1, 2]
                    
                    if fx > 0 and fy > 0 and cx > 0 and cy > 0:
                        logger.info(f"  ✓ Sample intrinsics look reasonable: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                    else:
                        warning_msg = f"Sample intrinsics may be unusual: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
                        validation_results['errors'].append(warning_msg)
                        logger.warning(f"  ⚠ {warning_msg}")
                else:
                    error_msg = f"Intrinsic matrices have wrong shape: {intrinsic.shape[1:]}, expected (3, 3)"
                    validation_results['errors'].append(error_msg)
                    validation_results['validation_passed'] = False
                    logger.error(f"  ✗ {error_msg}")
                
                # Check extrinsic matrix format (should be 3x4)
                if extrinsic.shape[1:] == (3, 4):
                    logger.info(f"  ✓ Extrinsic matrices have correct shape (3x4)")
                else:
                    error_msg = f"Extrinsic matrices have wrong shape: {extrinsic.shape[1:]}, expected (3, 4)"
                    validation_results['errors'].append(error_msg)
                    validation_results['validation_passed'] = False
                    logger.error(f"  ✗ {error_msg}")
                
                # Check point cloud dimensions
                expected_height, expected_width = 518, 518  # Based on metadata analysis
                if points3d.shape[1:3] == (expected_height, expected_width):
                    logger.info(f"  ✓ Point cloud has expected spatial resolution: {expected_height}x{expected_width}")
                else:
                    warning_msg = f"Point cloud spatial resolution: {points3d.shape[1:3]}, expected ({expected_height}, {expected_width})"
                    logger.warning(f"  ⚠ {warning_msg}")
                
            except Exception as e:
                error_msg = f"Failed consistency validation: {e}"
                validation_results['errors'].append(error_msg)
                validation_results['validation_passed'] = False
                logger.error(f"  ✗ {error_msg}")
    
    except Exception as e:
        validation_results['errors'].append(f"General validation error: {e}")
        validation_results['validation_passed'] = False
        logger.error(f"✗ General validation error: {e}")
    
    return validation_results


def validate_images_directory(images_path: Path, segment_name: str, expected_frames: int, logger):
    """Validate corresponding images directory."""
    logger.info(f"Validating images for segment: {segment_name}")
    
    validation_results = {
        'segment_name': segment_name,
        'images_directory_exists': False,
        'image_count': 0,
        'expected_frames': expected_frames,
        'images_match_frames': False,
        'sample_images': [],
        'validation_passed': True,
        'errors': []
    }
    
    segment_image_dir = images_path / segment_name
    
    if segment_image_dir.exists():
        validation_results['images_directory_exists'] = True
        
        # Count images
        image_files = list(segment_image_dir.glob('*.jpg')) + list(segment_image_dir.glob('*.png'))
        validation_results['image_count'] = len(image_files)
        
        logger.info(f"  Found {len(image_files)} images")
        
        # Check if image count matches frame count
        if len(image_files) == expected_frames:
            validation_results['images_match_frames'] = True
            logger.info(f"  ✓ Image count matches frame count: {expected_frames}")
        elif abs(len(image_files) - expected_frames) <= 1:
            # Allow for off-by-one differences
            validation_results['images_match_frames'] = True
            logger.info(f"  ✓ Image count approximately matches frame count: {len(image_files)} vs {expected_frames}")
        else:
            error_msg = f"Image count mismatch: {len(image_files)} images, {expected_frames} frames"
            validation_results['errors'].append(error_msg)
            validation_results['validation_passed'] = False
            logger.error(f"  ✗ {error_msg}")
        
        # Sample a few images to check they exist and have reasonable sizes
        sample_count = min(5, len(image_files))
        for i, img_file in enumerate(image_files[:sample_count]):
            try:
                file_size = img_file.stat().st_size
                validation_results['sample_images'].append({
                    'filename': img_file.name,
                    'size_bytes': file_size,
                    'size_mb': file_size / (1024 * 1024)
                })
                logger.info(f"  ○ Sample image {img_file.name}: {file_size / (1024 * 1024):.2f} MB")
            except Exception as e:
                error_msg = f"Failed to read image {img_file.name}: {e}"
                validation_results['errors'].append(error_msg)
                logger.warning(f"  ⚠ {error_msg}")
        
    else:
        validation_results['validation_passed'] = False
        error_msg = f"Images directory does not exist: {segment_image_dir}"
        validation_results['errors'].append(error_msg)
        logger.error(f"  ✗ {error_msg}")
    
    return validation_results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate custom point cloud data")
    parser.add_argument("--custom_pc_path", type=str, required=True,
                       help="Path to custom point cloud data directory")
    parser.add_argument("--images", type=str, default=None,
                       help="Path to images directory")
    parser.add_argument("--output_report", type=str, default="validation_report.json",
                       help="Output file for validation report")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("CUSTOM POINT CLOUD DATA VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Custom PC path: {args.custom_pc_path}")
    logger.info(f"Images path: {args.images}")
    
    custom_pc_path = Path(args.custom_pc_path)
    images_path = Path(args.images) if args.images else None
    
    # Check if paths exist
    if not custom_pc_path.exists():
        logger.error(f"Custom point cloud path does not exist: {custom_pc_path}")
        return 1
    
    if images_path and not images_path.exists():
        logger.error(f"Images path does not exist: {images_path}")
        return 1
    
    # Find segment directories
    segment_dirs = []
    for item in custom_pc_path.iterdir():
        if item.is_dir() and item.name.startswith('segment-'):
            segment_dirs.append(item)
    
    if not segment_dirs:
        logger.error("No segment directories found!")
        return 1
    
    logger.info(f"Found {len(segment_dirs)} segment directories")
    
    # Validate each segment
    validation_report = {
        'total_segments': len(segment_dirs),
        'segments_passed': 0,
        'segments_failed': 0,
        'segment_results': [],
        'image_results': [],
        'overall_passed': True,
        'summary': {}
    }
    
    for segment_dir in segment_dirs:
        logger.info("-" * 40)
        
        # Validate segment data
        segment_result = validate_segment_data(segment_dir, logger)
        validation_report['segment_results'].append(segment_result)
        
        if segment_result['validation_passed']:
            validation_report['segments_passed'] += 1
        else:
            validation_report['segments_failed'] += 1
            validation_report['overall_passed'] = False
        
        # Validate corresponding images if provided
        if images_path and 'frame_count' in segment_result:
            image_result = validate_images_directory(
                images_path, 
                segment_dir.name, 
                segment_result['frame_count'], 
                logger
            )
            validation_report['image_results'].append(image_result)
            
            if not image_result['validation_passed']:
                validation_report['overall_passed'] = False
    
    # Generate summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    if validation_report['overall_passed']:
        logger.info("🎉 All validations PASSED!")
        logger.info(f"  ✓ {validation_report['segments_passed']} segments validated successfully")
        
        # Calculate total statistics
        total_points = 0
        total_frames = 0
        for result in validation_report['segment_results']:
            if 'frame_count' in result:
                total_frames += result['frame_count']
            if 'data_shapes' in result and 'points3d_unproj.npy' in result['data_shapes']:
                shape = result['data_shapes']['points3d_unproj.npy']
                total_points += shape[0] * shape[1] * shape[2]  # frames * height * width
        
        validation_report['summary'] = {
            'total_frames': total_frames,
            'estimated_total_points': total_points,
            'segments_validated': validation_report['segments_passed']
        }
        
        logger.info(f"  ✓ Total frames across all segments: {total_frames}")
        logger.info(f"  ✓ Estimated total 3D points: {total_points:,}")
        
        if images_path:
            total_images = sum(r['image_count'] for r in validation_report['image_results'])
            logger.info(f"  ✓ Total images found: {total_images}")
    else:
        logger.error("❌ Validation FAILED!")
        logger.error(f"  ✓ Segments passed: {validation_report['segments_passed']}")
        logger.error(f"  ✗ Segments failed: {validation_report['segments_failed']}")
        
        # List errors
        logger.error("\nErrors found:")
        for result in validation_report['segment_results']:
            if result['errors']:
                logger.error(f"  {result['segment_name']}:")
                for error in result['errors']:
                    logger.error(f"    - {error}")
    
    # Save validation report
    with open(args.output_report, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"\nDetailed validation report saved to: {args.output_report}")
    
    return 0 if validation_report['overall_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())