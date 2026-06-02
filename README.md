# kitti_perception

This is a small camera-LiDAR perception project built on the KITTI raw dataset. KITTI is a widely used autonomous driving dataset collected from a vehicle equipped with multiple sensors, including four cameras, a Velodyne 3D LiDAR scanner, and a GPS/IMU localization unit. The camera setup includes one grayscale stereo pair and one RGB stereo pair; in the raw dataset, cam0 and cam1 are grayscale cameras, while cam2 and cam3 are the left and right RGB cameras. The cameras are synchronized at around 10 Hz with the Velodyne scanner, and the camera trigger is aligned so that images are captured when the LiDAR is roughly facing forward. Velodyne scans are stored as binary point clouds in the format [x, y, z, reflectance], where the coordinates are in meters in the Velodyne frame. KITTI also provides calibration files that define the relationship between the camera, LiDAR, and GPS/IMU coordinate systems. In KITTI, the camera frame is defined as x = right, y = down, z = forward, while the Velodyne frame is x = forward, y = left, z = up.

In this project, I used the KITTI raw data to build a step-by-step geometric perception pipeline. I inspected the dataset structure and calibration files, worked with the intrinsic and extrinsic matrices, transformed LiDAR points from the Velodyne frame into the camera frame, and projected them onto the cam2 RGB image. After validating the projection, I generated sparse depth maps and camera-LiDAR depth overlays. I also converted the LiDAR point cloud into Bird’s Eye View (BEV) maps, including occupancy, density, height, and intensity representations. BEV is useful because it converts the 3D scene into a top-down metric grid, making distances, road layout, point density, and object positions easier to reason about. I also tested ground plane estimation using RANSAC to separate road-like ground points from above-ground and below-ground residual points. The goal of this project is not to train a black-box model, but to understand the core geometry behind camera-LiDAR sensor fusion and build a clean perception front-end from raw autonomous driving sensor data.

KITTI dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php

<img width="475" height="340" alt="Camera LiDAR Projection" src="https://github.com/user-attachments/assets/a8f7bca0-23b8-44e2-b2f8-df75bd07366d" />

LiDAR points projected onto the cam2 RGB image using KITTI calibration.

<img width="981" height="433" alt="BEV and Camera Visualization" src="https://github.com/user-attachments/assets/b7a31a2e-2ac5-478a-9c88-19fccc88f83f" />

Camera-LiDAR visualization with BEV feature maps and RANSAC-based ground modeling.

## Pipeline

text KITTI raw data    -> camera / LiDAR calibration    -> Velodyne point cloud loading    -> LiDAR-to-camera transformation    -> projection onto RGB image    -> sparse depth map generation    -> BEV feature map generation    -> RANSAC ground plane estimation    -> camera + BEV visualization 

## Key scripts

text scripts/   01_inspect_dataset.py   02_inspect_calibration.py   03_transform_single_point.py   04_project_single_point.py   05_project_all_points.py   06_sparse_depth_map.py   07_sparse_depth_map_video.py   08_camera_depth_overlay_video.py   09_1_bev_coordinates.py   09_2_bev_ocuppancy_map.py   09_3_bev_density_map.py   09_4_bev_height_map.py   09_5_bev_height_map_video.py   10_all_views_single_video.py   11_ransac_ground_plane_single_frame.py   12_ransac_ground_camera_bev_video.py   13_below_surface_clusters_video.py 

## Notes

This project uses pykitti to load KITTI raw data, calibration files, camera images, timestamps, and Velodyne point clouds. A similar idea could later be extended to a custom loader for my own 3D scanner data, similar to a lightweight pyeagle or pymaker package. That would make it easier to parse custom LiDAR frames, fisheye camera images, calibration matrices, timestamps, and sensor transforms in the same structured way that pykitti handles KITTI.
