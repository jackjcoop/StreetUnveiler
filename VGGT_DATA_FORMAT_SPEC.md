# VGGT Data Format Specification

## Document Overview

This document provides a detailed technical specification of the VGGT (Video-Guided Generation of Textures) point cloud data format used in StreetUnveiler's custom point cloud integration.

## Data Analysis Summary

Based on analysis of the sample data in `new_data/WOD_points/segment-10017090168044687777_6380_000_6400_000_with_camera_labels/`, the following structure has been identified:

### File Inventory

| Filename | Size | Purpose | Required |
|----------|------|---------|----------|
| `points3d_unproj.npy` | 1.2 GB | 3D point coordinates | ✅ Yes |
| `intrinsic.npy` | 7.1 KB | Camera intrinsic parameters | ✅ Yes |
| `extrinsic.npy` | 9.4 KB | Camera extrinsic parameters | ✅ Yes |
| `depth_map.npy` | 202.7 MB | Per-pixel depth values | ❌ Optional |
| `point_map.npy` | 608.0 MB | Point mappings | ❌ Optional |
| `depth_conf.npy` | 202.7 MB | Depth confidence scores | ❌ Optional |
| `point_conf.npy` | 202.7 MB | Point confidence scores | ❌ Optional |

### Corresponding Images

- **Location**: `new_data/WOD_undistorted_images/segment-10017090168044687777_6380_000_6400_000_with_camera_labels/`
- **Count**: 198 images (matching frame count)
- **Format**: JPEG (`.jpg`)
- **Naming**: Sequential (`00000.jpg`, `00001.jpg`, ..., `00197.jpg`)

## Detailed Data Specifications

### 1. points3d_unproj.npy

**Purpose**: Contains 3D world coordinates for each pixel in the image sequence.

**Specifications**:
- **Data Type**: `float64`
- **Shape**: `(198, 518, 518, 3)`
  - 198 frames (temporal dimension)
  - 518x518 spatial resolution
  - 3 coordinates (X, Y, Z)
- **Value Range**: [-0.486, 2.458] meters
- **Statistics**:
  - Mean: 0.327 m
  - Std Dev: 0.572 m
  - Median: 0.059 m
- **Coordinate System**: Camera coordinate frame
- **Memory Usage**: ~1.2 GB

**Interpretation**:
- Each `(i, j)` pixel in frame `f` has a 3D coordinate `(x, y, z)`
- Coordinates represent the 3D location of the surface point visible at that pixel
- Invalid/infinite depth points typically have special values or NaN

### 2. intrinsic.npy

**Purpose**: Camera intrinsic parameters for each frame.

**Specifications**:
- **Data Type**: `float32`
- **Shape**: `(198, 3, 3)`
- **Format**: Standard pinhole camera model
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```
- **Value Range**: [0.0, 615.3] pixels
- **Sample Values**:
  - First frame: fx=579.7, fy=572.0, cx=cy=259.0
  - Last frame: fx=576.4, fy=573.4, cx=cy=259.0

**Interpretation**:
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point coordinates (image center)
- Small variations across frames indicate camera calibration changes

### 3. extrinsic.npy

**Purpose**: Camera extrinsic parameters (pose) for each frame.

**Specifications**:
- **Data Type**: `float32`
- **Shape**: `(198, 3, 4)`
- **Format**: `[R|t]` matrix where:
  - `R`: 3x3 rotation matrix (world → camera)
  - `t`: 3x1 translation vector (world → camera)
- **Value Range**: [-1.601, 1.0]

**Sample Values**:
```
First frame (identity-like):
[[ 1.0000  -0.0001   0.0001]
 [ 0.0001   1.0000  -0.0000]
 [-0.0001   0.0000   1.0000]]

Last frame (rotated):
[[ 0.9873  -0.0081  -0.1585]
 [ 0.0204   0.9969   0.0759]
 [ 0.1574  -0.0782   0.9844]]
```

**Interpretation**:
- Represents camera trajectory through the scene
- First frame is nearly identity (reference pose)
- Gradual rotation/translation indicates smooth camera motion

### 4. depth_map.npy (Optional)

**Specifications**:
- **Data Type**: `float32`
- **Shape**: `(198, 518, 518, 1)`
- **Value Range**: [0.036, 1.871] meters
- **Statistics**:
  - Mean: 0.248 m
  - Std Dev: 0.186 m
  - Median: 0.187 m

**Interpretation**:
- Per-pixel depth values in meters
- Represents distance from camera to surface
- Related to but distinct from `points3d_unproj` (which gives full 3D coordinates)

### 5. point_map.npy (Optional)

**Specifications**:
- **Data Type**: `float32`
- **Shape**: `(198, 518, 518, 3)`
- **Value Range**: [-0.885, 3.613]
- **Size**: 608 MB

**Interpretation**:
- Alternative representation of 3D points
- Possibly in a different coordinate system or processed format
- Larger file size suggests float32 vs float64 difference with points3d_unproj

### 6. Confidence Maps (Optional)

**depth_conf.npy & point_conf.npy**:
- **Data Type**: `float32`
- **Shape**: `(198, 518, 518, 1)` or `(198, 518, 518)`
- **Value Range**: [0.0, 1.0] (confidence scores)
- **Purpose**: Quality/reliability scores for depth/point estimates

## Coordinate System Conventions

### Camera Coordinate System
- **Origin**: Camera optical center
- **X-axis**: Right (image +X direction)
- **Y-axis**: Down (image +Y direction)  
- **Z-axis**: Forward (into the scene)
- **Units**: Meters

### Image Coordinate System
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Right (columns)
- **Y-axis**: Down (rows)
- **Resolution**: 518 × 518 pixels
- **Units**: Pixels

### World Coordinate System
- **Definition**: Defined by extrinsic matrices
- **Transformation**: `P_camera = R * P_world + t`
- **Units**: Meters

## Data Relationships

### Point Projection
The relationship between 3D points and image coordinates:

```
# 3D point in camera coordinates
P_cam = points3d_unproj[frame, row, col, :]

# Project to image coordinates
p_image = K @ P_cam
u = p_image[0] / p_image[2]  # should ≈ col
v = p_image[1] / p_image[2]  # should ≈ row
```

### Depth Consistency
```
# Depth should match point magnitude
depth_computed = ||points3d_unproj[frame, row, col, :]||
depth_provided = depth_map[frame, row, col, 0]
# These should be approximately equal
```

### Camera Motion
```
# Transform between frames
P_world = R[frame]^T @ (P_camera - t[frame])
```

## Data Quality Indicators

### Valid Data Identification
- **Depth range**: [0.036, 1.871] meters (reasonable for street scenes)
- **No NaN/Inf values**: Clean data confirmed
- **Consistent dimensions**: All arrays match expected frame count
- **Smooth camera motion**: Gradual changes in extrinsic parameters

### Potential Issues
- **Limited depth range**: Max depth ~1.87m may miss distant objects
- **Fixed resolution**: 518×518 may not match original image resolution
- **Coordinate system**: Verify camera coordinate conventions match expectations

## Usage Recommendations

### For Point Cloud Generation
1. Use `points3d_unproj.npy` as primary source
2. Filter using `point_conf.npy` if available (threshold ≥ 0.5)
3. Apply depth limits (e.g., 0.1m to 100m) for scene bounds
4. Downsample if memory constrained (>50K points per segment recommended)

### For Camera Calibration
1. Use `intrinsic.npy` for projection parameters
2. Use `extrinsic.npy` for camera poses
3. Verify coordinate system consistency with image data
4. Consider temporal variations in intrinsics

### Data Validation Checklist
- [ ] Frame counts match across all files
- [ ] Point coordinates are finite and reasonable
- [ ] Camera parameters produce valid projections
- [ ] Image-to-data correspondence is correct
- [ ] Coordinate systems are consistent

## Performance Notes

### Memory Requirements
- **Loading all data**: ~2.4 GB RAM per segment
- **Processing**: Additional 2-4x for intermediate arrays
- **Recommended**: 16GB+ system RAM for comfortable processing

### Optimization Strategies
- **Streaming**: Load and process one frame at a time
- **Confidence filtering**: Early elimination of low-confidence points
- **Spatial downsampling**: Reduce resolution if full quality not needed
- **Temporal sampling**: Skip frames for faster processing

---

This specification is based on analysis of the provided VGGT dataset and may need adjustment for other data sources or formats.