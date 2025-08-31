
#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

from scene.dataset_readers.colmap import readColmapSceneInfo
from scene.dataset_readers.blender import readNerfSyntheticInfo
from scene.dataset_readers.waymo import readWaymoInfo
from scene.dataset_readers.pandaset import readPandasetInfo
from scene.dataset_readers.kitti import readKittiInfo
from scene.dataset_readers.nuscenes import readNuScenesInfo
from scene.dataset_readers.custom import readCustomSceneInfo


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Waymo": readWaymoInfo,
    "Pandaset": readPandasetInfo,
    "Kitti": readKittiInfo,
    "NuScenes": readNuScenesInfo,
    "Custom": readCustomSceneInfo,
}

