# kitti_perception

A small camera-LiDAR perception project built with the KITTI raw dataset. KITTI provides synchronized camera images, Velodyne LiDAR scans, calibration files, and GPS/IMU data for autonomous driving research. In this project, I used cam2 RGB images and Velodyne point clouds `[x, y, z, reflectance]` to understand the core geometry behind sensor fusion: inspecting intrinsic and extrinsic calibration matrices, transforming LiDAR points from the Velodyne frame into the camera frame, projecting them onto the image plane, generating sparse depth overlays, building Bird’s Eye View (BEV) occupancy/density/height/intensity maps, and estimating the road ground plane with RANSAC. The goal is to build a clear geometric perception pipeline from raw sensor data rather than relying on a black-box model.

KITTI dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php

<img width="475" height="340" alt="Camera LiDAR Projection" src="https://github.com/user-attachments/assets/a8f7bca0-23b8-44e2-b2f8-df75bd07366d" />

*LiDAR points projected onto the cam2 RGB image using KITTI calibration.*

<img width="981" height="433" alt="BEV and Camera Visualization" src="https://github.com/user-attachments/assets/b7a31a2e-2ac5-478a-9c88-19fccc88f83f" />

*Camera-LiDAR visualization with BEV feature maps and RANSAC-based ground modeling.*


## Pipeline

```text
KITTI raw data
   -> calibration inspection
   -> LiDAR-to-camera projection
   -> sparse depth overlay
   -> BEV feature map generation
   -> RANSAC ground plane estimation
   -> camera + BEV visualization


scripts/
  01_inspect_dataset.py
  02_inspect_calibration.py
  03_transform_single_point.py
  04_project_single_point.py
  05_project_all_points.py
  06_sparse_depth_map.py
  07_sparse_depth_map_video.py
  08_camera_depth_overlay_video.py
  09_1_bev_coordinates.py
  09_2_bev_ocuppancy_map.py
  09_3_bev_density_map.py
  09_4_bev_height_map.py
  09_5_bev_height_map_video.py
  10_all_views_single_video.py
  11_ransac_ground_plane_single_frame.py
  12_ransac_ground_camera_bev_video.py
  13_below_surface_clusters_video.py
