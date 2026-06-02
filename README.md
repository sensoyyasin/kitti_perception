# kitti_perception

Small camera-LiDAR perception project using the KITTI raw dataset. KITTI provides synchronized camera images, Velodyne LiDAR scans, calibration files, and GPS/IMU data, which makes it useful for learning autonomous driving sensor fusion. In this project I used cam2/cam3 RGB cameras and Velodyne point clouds `[x, y, z, reflectance]` to inspect calibration, project LiDAR points into the camera image, generate sparse depth overlays, build BEV occupancy/density/height/intensity maps, and estimate the road ground plane with RANSAC.

KITTI dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php

<img width="475" height="340" alt="Camera LiDAR Projection" src="https://github.com/user-attachments/assets/a8f7bca0-23b8-44e2-b2f8-df75bd07366d" />

<img width="981" height="433" alt="BEV and Camera Visualization" src="https://github.com/user-attachments/assets/b7a31a2e-2ac5-478a-9c88-19fccc88f83f" />
