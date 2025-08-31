#!/usr/bin/env python3
"""
Basic file structure validation for custom VGGT point cloud data.
No external dependencies required.

Usage:
    python basic_data_check.py --custom_pc_path ./new_data/WOD_points --images ./new_data/WOD_undistorted_images
"""

import os
import sys
import argparse
import json
from pathlib import Path


def check_segment_files(segment_dir):
    """Check if all required files exist in a segment directory."""
    required_files = [
        'points3d_unproj.npy',
        'intrinsic.npy',
        'extrinsic.npy'
    ]
    
    optional_files = [
        'depth_map.npy',
        'point_map.npy',
        'depth_conf.npy',
        'point_conf.npy'
    ]
    
    result = {
        'segment_name': segment_dir.name,
        'required_files_found': [],
        'required_files_missing': [],
        'optional_files_found': [],
        'file_sizes': {},
        'validation_passed': True
    }
    
    # Check required files
    for filename in required_files:
        filepath = segment_dir / filename
        if filepath.exists():
            result['required_files_found'].append(filename)
            result['file_sizes'][filename] = filepath.stat().st_size
        else:
            result['required_files_missing'].append(filename)
            result['validation_passed'] = False
    
    # Check optional files
    for filename in optional_files:
        filepath = segment_dir / filename
        if filepath.exists():
            result['optional_files_found'].append(filename)
            result['file_sizes'][filename] = filepath.stat().st_size
    
    return result


def check_images(images_path, segment_name):
    """Check if images directory exists and count images."""
    result = {
        'segment_name': segment_name,
        'directory_exists': False,
        'image_count': 0,
        'image_extensions': [],
        'sample_files': []
    }
    
    if images_path is None:
        return result
    
    segment_image_dir = images_path / segment_name
    if segment_image_dir.exists():
        result['directory_exists'] = True
        
        # Count different image types
        jpg_files = list(segment_image_dir.glob('*.jpg'))
        png_files = list(segment_image_dir.glob('*.png'))
        jpeg_files = list(segment_image_dir.glob('*.jpeg'))
        
        result['image_count'] = len(jpg_files) + len(png_files) + len(jpeg_files)
        
        if jpg_files:
            result['image_extensions'].append('jpg')
        if png_files:
            result['image_extensions'].append('png')
        if jpeg_files:
            result['image_extensions'].append('jpeg')
        
        # Get sample filenames
        all_images = jpg_files + png_files + jpeg_files
        result['sample_files'] = [f.name for f in all_images[:5]]  # First 5 files
    
    return result


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Basic validation of custom point cloud data structure")
    parser.add_argument("--custom_pc_path", type=str, required=True,
                       help="Path to custom point cloud data directory")
    parser.add_argument("--images", type=str, default=None,
                       help="Path to images directory")
    parser.add_argument("--output", type=str, default="data_check_report.json",
                       help="Output file for validation report")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BASIC DATA STRUCTURE VALIDATION")
    print("=" * 60)
    print(f"Custom PC path: {args.custom_pc_path}")
    print(f"Images path: {args.images}")
    print()
    
    custom_pc_path = Path(args.custom_pc_path)
    images_path = Path(args.images) if args.images else None
    
    # Check if paths exist
    if not custom_pc_path.exists():
        print(f"[ERROR] Custom point cloud path does not exist: {custom_pc_path}")
        return 1
    
    if images_path and not images_path.exists():
        print(f"[ERROR] Images path does not exist: {images_path}")
        return 1
    
    # Find segment directories
    segment_dirs = []
    for item in custom_pc_path.iterdir():
        if item.is_dir() and item.name.startswith('segment-'):
            segment_dirs.append(item)
    
    if not segment_dirs:
        print("[ERROR] No segment directories found!")
        print("Expected directories starting with 'segment-'")
        return 1
    
    print(f"[OK] Found {len(segment_dirs)} segment directories")
    print()
    
    # Validate each segment
    report = {
        'validation_timestamp': str(Path.cwd()),  # Simple timestamp substitute
        'total_segments': len(segment_dirs),
        'segments_passed': 0,
        'segments_failed': 0,
        'segment_results': [],
        'image_results': [],
        'overall_status': 'PASSED'
    }
    
    for i, segment_dir in enumerate(segment_dirs, 1):
        print(f"[{i}/{len(segment_dirs)}] Checking segment: {segment_dir.name}")
        
        # Check segment files
        segment_result = check_segment_files(segment_dir)
        report['segment_results'].append(segment_result)
        
        if segment_result['validation_passed']:
            print("  [OK] All required files found")
            report['segments_passed'] += 1
        else:
            print("  [ERROR] Missing required files:")
            for missing in segment_result['required_files_missing']:
                print(f"    - {missing}")
            report['segments_failed'] += 1
            report['overall_status'] = 'FAILED'
        
        # Show file sizes
        print(f"  Files found:")
        for filename, size in segment_result['file_sizes'].items():
            status = "[REQ]" if filename in segment_result['required_files_found'] else "[OPT]"
            print(f"    {status} {filename}: {format_file_size(size)}")
        
        if segment_result['optional_files_found']:
            print(f"  Optional files: {', '.join(segment_result['optional_files_found'])}")
        
        # Check images
        if images_path:
            image_result = check_images(images_path, segment_dir.name)
            report['image_results'].append(image_result)
            
            if image_result['directory_exists']:
                print(f"  Images: {image_result['image_count']} files ({', '.join(image_result['image_extensions']) if image_result['image_extensions'] else 'none'})")
                if image_result['sample_files']:
                    print(f"    Sample: {', '.join(image_result['sample_files'][:3])}...")
            else:
                print(f"  [WARNING] No images directory found")
        
        print()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if report['overall_status'] == 'PASSED':
        print("[SUCCESS] VALIDATION PASSED!")
        print(f"[OK] {report['segments_passed']} segments validated successfully")
        
        # Calculate totals
        total_size = 0
        total_images = 0
        
        for result in report['segment_results']:
            for size in result['file_sizes'].values():
                total_size += size
        
        for result in report['image_results']:
            total_images += result['image_count']
        
        print(f"Total data size: {format_file_size(total_size)}")
        if images_path:
            print(f"Total images: {total_images}")
        
        print("\n[SUCCESS] The custom point cloud data appears to be correctly structured!")
        print("[SUCCESS] Ready for integration with StreetUnveiler!")
        
    else:
        print("[FAILED] VALIDATION FAILED!")
        print(f"[OK] Segments passed: {report['segments_passed']}")
        print(f"[ERROR] Segments failed: {report['segments_failed']}")
        print("\n[ERROR] Please fix the missing files before proceeding.")
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {args.output}")
    
    return 0 if report['overall_status'] == 'PASSED' else 1


if __name__ == "__main__":
    sys.exit(main())