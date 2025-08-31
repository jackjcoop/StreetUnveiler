# Custom VGGT Point Cloud Integration for StreetUnveiler

## Overview

This document describes the integration of custom VGGT-derived point clouds into the StreetUnveiler 3D Gaussian Splatting pipeline. The implementation allows you to use custom point cloud data as an alternative or supplement to the default COLMAP/LiDAR initialization.

## Table of Contents

1. [Features](#features)
2. [Installation & Setup](#installation--setup)
3. [Data Format Specification](#data-format-specification)
4. [Usage Guide](#usage-guide)
5. [Command-Line Arguments](#command-line-arguments)
6. [Example Invocations](#example-invocations)
7. [Validation Tools](#validation-tools)
8. [Technical Implementation](#technical-implementation)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)

## Features

- **Custom Point Cloud Support**: Load point clouds from VGGT-derived data formats
- **Flexible Initialization Modes**:
  - `replace`: Use custom point cloud instead of COLMAP/LiDAR
  - `augment`: Combine custom point cloud with existing sources
- **Automatic Data Validation**: Built-in validation for data consistency
- **Camera Parameter Integration**: Full support for intrinsic/extrinsic camera parameters
- **Confidence-Based Filtering**: Filter points based on confidence scores
- **Memory Efficient**: Configurable point cloud downsampling
- **Backward Compatible**: Existing workflows continue to work unchanged

## Installation & Setup

### Prerequisites

Ensure you have StreetUnveiler installed with all standard dependencies:
- PyTorch
- NumPy
- OpenCV
- Other standard StreetUnveiler dependencies (see `requirements.txt`)

### File Structure Requirements

Your custom point cloud data should be organized as follows:

```
custom_pc_path/
├── segment-<ID1>/
│   ├── points3d_unproj.npy      # Required: 3D points (frames, height, width, 3)
│   ├── intrinsic.npy            # Required: Camera intrinsics (frames, 3, 3)
│   ├── extrinsic.npy            # Required: Camera extrinsics (frames, 3, 4)
│   ├── depth_map.npy            # Optional: Depth maps
│   ├── point_map.npy            # Optional: Point maps
│   ├── depth_conf.npy           # Optional: Depth confidence
│   └── point_conf.npy           # Optional: Point confidence
├── segment-<ID2>/
│   └── ... (same structure)
└── ...

images_path/                     # Optional: Corresponding images
├── segment-<ID1>/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── segment-<ID2>/
│   └── ...
└── ...
```

## Data Format Specification

### Required Files

#### `points3d_unproj.npy`
- **Shape**: `(n_frames, height, width, 3)`
- **Type**: `float64` or `float32`
- **Content**: 3D coordinates for each pixel in camera frame
- **Units**: Meters
- **Coordinate System**: Camera coordinate system (typically: +X right, +Y down, +Z forward)

#### `intrinsic.npy`
- **Shape**: `(n_frames, 3, 3)`
- **Type**: `float32`
- **Content**: Camera intrinsic matrices in standard format:
  ```
  [[fx,  0, cx],
   [ 0, fy, cy],
   [ 0,  0,  1]]
  ```
- **Units**: Pixels

#### `extrinsic.npy`
- **Shape**: `(n_frames, 3, 4)`
- **Type**: `float32`
- **Content**: Camera extrinsic matrices `[R|t]` where:
  - `R`: 3x3 rotation matrix (world to camera)
  - `t`: 3x1 translation vector (world to camera)
- **Units**: Meters (for translation)

### Optional Files

#### `depth_map.npy`
- **Shape**: `(n_frames, height, width, 1)`
- **Type**: `float32`
- **Content**: Depth values for each pixel
- **Units**: Meters

#### `point_conf.npy` / `depth_conf.npy`
- **Shape**: `(n_frames, height, width, 1)` or `(n_frames, height, width)`
- **Type**: `float32`
- **Content**: Confidence scores (0.0 to 1.0)
- **Usage**: Used for filtering low-confidence points

### Coordinate Systems

- **Camera Coordinates**: Right-handed system with +Z forward
- **World Coordinates**: Defined by extrinsic matrices
- **Image Coordinates**: Origin at top-left, +X right, +Y down

## Usage Guide

### Basic Usage

1. **Validate Your Data**:
   ```bash
   python basic_data_check.py --custom_pc_path ./path/to/WOD_points --images ./path/to/images
   ```

2. **Run Training with Custom Point Cloud**:
   ```bash
   python train.py \
       --custom_pc_path ./new_data/WOD_points \
       --images ./new_data/WOD_undistorted_images \
       --use_custom_init \
       --custom_init_mode replace \
       --model_path ./output/custom_experiment
   ```

### Advanced Usage

- **Combine with Existing Data**:
  ```bash
  python train.py \
      --source_path /path/to/lidar \
      --colmap_path /path/to/colmap \
      --custom_pc_path ./new_data/WOD_points \
      --use_custom_init \
      --custom_init_mode augment
  ```

- **Tune Point Cloud Parameters**:
  ```bash
  python train.py \
      --custom_pc_path ./new_data/WOD_points \
      --use_custom_init \
      --custom_confidence_threshold 0.7 \
      --custom_max_points_per_segment 100000
  ```

## Command-Line Arguments

### Custom Point Cloud Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--custom_pc_path` | str | `""` | Path to custom point cloud data directory |
| `--use_custom_init` | bool | `False` | Enable custom point cloud initialization |
| `--custom_init_mode` | str | `"replace"` | Mode: `"replace"` or `"augment"` |
| `--custom_confidence_threshold` | float | `0.5` | Minimum confidence for point inclusion |
| `--custom_max_points_per_segment` | int | `50000` | Maximum points per segment (for memory management) |

### Standard StreetUnveiler Arguments

All standard StreetUnveiler arguments continue to work:
- `--source_path`: LiDAR data path
- `--colmap_path`: COLMAP data path  
- `--images`: Images directory path
- `--model_path`: Output model path
- `--resolution`: Image resolution
- `--white_background`: Use white background
- etc.

## Example Invocations

### 1. Replace COLMAP with Custom Point Cloud

```bash
python train.py \
    --source_path /path/to/lidar \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images \
    --use_custom_init \
    --custom_init_mode replace \
    --model_path ./output/custom_only
```

### 2. Augment LiDAR with Custom Points

```bash
python train.py \
    --source_path /path/to/lidar \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images \
    --use_custom_init \
    --custom_init_mode augment \
    --model_path ./output/lidar_plus_custom
```

### 3. Use Only Custom Point Cloud (No LiDAR/COLMAP)

```bash
python train.py \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images \
    --use_custom_init \
    --custom_init_mode replace \
    --model_path ./output/custom_only
```

### 4. High-Quality Settings

```bash
python train.py \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images \
    --use_custom_init \
    --custom_init_mode replace \
    --custom_confidence_threshold 0.8 \
    --custom_max_points_per_segment 200000 \
    --model_path ./output/high_quality \
    --iterations 100000 \
    --resolution 2
```

## Validation Tools

### Basic Data Structure Validation

```bash
python basic_data_check.py \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images \
    --output validation_report.json
```

This tool checks:
- Required file presence
- File sizes and formats
- Image-to-frame count matching
- Basic data structure integrity

### Advanced Validation (Requires PyTorch)

```bash
python test_custom_pointcloud.py \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images
```

This tool performs:
- Full data loading tests
- Point cloud and camera validation  
- Scene integration testing
- Memory usage analysis

### Visualization Tools (Requires PyTorch + matplotlib)

```bash
python visualize_custom_pointcloud.py \
    --custom_pc_path ./new_data/WOD_points \
    --images ./new_data/WOD_undistorted_images \
    --output_dir ./visualization_output
```

Creates:
- 3D point cloud visualizations
- Camera pose plots
- Data statistics charts
- Projection examples

## Technical Implementation

### Modified Files

1. **`arguments/__init__.py`**: Added custom point cloud command-line arguments
2. **`scene/__init__.py`**: Modified Scene class to support custom initialization
3. **`scene/dataset_readers/__init__.py`**: Added custom reader to callback registry
4. **`scene/dataset_readers/custom.py`**: Custom dataset reader implementation
5. **`scene/custom_pointcloud_loader.py`**: Core point cloud loading functionality

### New Files

1. **`test_custom_pointcloud.py`**: Comprehensive integration test suite
2. **`visualize_custom_pointcloud.py`**: Visualization tools
3. **`basic_data_check.py`**: Basic validation tool
4. **`validate_custom_data.py`**: Advanced validation (requires NumPy)

### Integration Points

- **Scene Initialization**: Custom initialization is checked before standard methods
- **Point Cloud Loading**: Uses `SemanticPointCloud` for compatibility
- **Camera Creation**: Converts custom data to StreetUnveiler `Camera` objects
- **Gaussian Initialization**: Integrates seamlessly with existing Gaussian creation

### Memory Management

- **Confidence Filtering**: Removes low-confidence points early
- **Downsampling**: Configurable maximum points per segment
- **Batch Processing**: Processes segments individually to manage memory
- **Temporary Files**: Uses temporary PLY files for compatibility

## Troubleshooting

### Common Issues

1. **"No segment directories found!"**
   - Ensure directories start with `segment-`
   - Check path spelling and accessibility

2. **"Missing required files"**
   - Verify all required `.npy` files exist
   - Check file permissions

3. **"Frame count mismatch"**
   - Ensure all required files have same number of frames
   - Check data consistency across files

4. **Memory errors during loading**
   - Reduce `--custom_max_points_per_segment`
   - Increase `--custom_confidence_threshold`
   - Process fewer segments at once

5. **Camera parameters look wrong**
   - Verify coordinate system conventions
   - Check intrinsic matrix format
   - Validate extrinsic matrix transformations

### Debug Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check intermediate results**:
   - Use validation tools before training
   - Inspect temporary PLY files
   - Verify camera poses make sense

3. **Compare with known datasets**:
   - Test with COLMAP data first
   - Compare point cloud distributions
   - Check camera trajectories

## Performance Considerations

### Data Size Management

- **Typical data sizes**: 1-2 GB per segment
- **Memory usage**: ~4x file size during loading
- **Recommended RAM**: 16GB+ for large datasets

### Optimization Settings

- **For speed**: Lower confidence threshold, fewer points per segment
- **For quality**: Higher confidence threshold, more points per segment
- **For memory**: Process segments individually, use temporary files

### Hardware Recommendations

- **GPU**: CUDA-capable GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: Fast SSD for data loading

## Contact & Support

For questions or issues related to the custom point cloud integration:

1. Check this documentation first
2. Run validation tools to identify issues
3. Review log files for error details
4. Test with provided validation scripts

## Changelog

### v1.0 - Initial Implementation
- Custom VGGT point cloud support
- Replace and augment modes
- Confidence-based filtering
- Comprehensive validation tools
- Full documentation

---

**Note**: This integration maintains full backward compatibility with existing StreetUnveiler workflows. All standard functionality continues to work exactly as before.