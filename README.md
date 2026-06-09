# KITTI Perception Pipeline

An end-to-end LiDAR-camera fusion and Bird's Eye-View (BEV) vehicle segmentation pipeline built on the KITTI autonomous driving dataset.

## Project Summary
 
This project explores the full perception stack of a self-driving vehicle using the KITTI Raw and Object Detection datasets. Starting from raw sensor data — a 64-beam Velodyne LiDAR and a stereo camera pair — the pipeline progresses through calibration, sensor fusion, classical road detection, 3-D object annotation, and finally trains a deep learning model to segment vehicles in BEV space.
 
The codebase is structured as a numbered sequence of runnable scripts (01 → 15), each building on the previous stage. Every script delegates to a matching class in `scripts/classes/`, keeping the scripts themselves concise entry points and the logic reusable.
 
**Key capabilities:**
- LiDAR-to-camera projection using calibrated intrinsic and extrinsic matrices
- Sparse and dense depth-map generation with video export
- BEV feature maps: occupancy, density, height, and intensity channels
- RANSAC-based ground-plane estimation and above/below-ground point separation
- Road-corridor detection via LiDAR clustering and SegFormer camera segmentation, fused into a single mask
- 3-D bounding-box visualisation on both the camera image and BEV plane
- Automated BEV ground-truth mask generation from KITTI object labels
- Training and evaluation of Small U-Net and ResNet segmentation models on BEV inputs
---


<img width="500" height="500" alt="Camera LiDAR Projection" src="https://github.com/user-attachments/assets/a8f7bca0-23b8-44e2-b2f8-df75bd07366d" />

<img width="981" height="433" alt="BEV and Camera Visualization" src="https://github.com/user-attachments/assets/b7a31a2e-2ac5-478a-9c88-19fccc88f83f" />

In this project, I used the KITTI raw data to build a step-by-step geometric perception pipeline. I inspected the dataset structure and calibration files, worked with the intrinsic and extrinsic matrices, transformed LiDAR points from the Velodyne frame into the camera frame, and projected them onto the cam2 RGB image. After validating the projection, I generated sparse depth maps and camera-LiDAR depth overlays. I also converted the LiDAR point cloud into Bird’s Eye View (BEV) maps, including occupancy, density, height, and intensity representations. BEV is useful because it converts the 3D scene into a top-down metric grid, making distances, road layout, point density, and object positions easier to reason about. I also tested ground plane estimation using RANSAC to separate road-like ground points from above-ground and below-ground residual points. The goal of this project is not to train a black-box model, but to understand the core geometry behind camera-LiDAR sensor fusion and build a clean perception front-end from raw autonomous driving sensor data.

## Repository Structure
 
```
kitti_perception_pipeline/
├── config/
│   ├── config.py          # Loads paths from .env; falls back to sensible defaults
│   └── .env               # (not committed) KITTI_RAW_DIR, KITTI_OBJ_DIR, etc.
├── scripts/
│   ├── bootstrap.py       # Adds project root to sys.path
│   ├── 01_inspect_dataset.py
│   ├── 02_inspect_calibration.py
│   ├── 03_transform_single_point.py
│   ├── 04_project_single_point.py
│   ├── 05_project_all_points.py
│   ├── 06_sparse_depth_map.py
│   ├── 07_sparse_depth_map_video.py
│   ├── 08_camera_depth_overlay_video.py
│   ├── 09_1_bev_coordinates.py
│   ├── 09_2_bev_occupancy_map.py
│   ├── 09_3_bev_density_map.py
│   ├── 09_4_bev_height_map.py
│   ├── 09_5_bev_feature_maps_video.py
│   ├── 10_all_views_single_video.py
│   ├── 11_ransac_ground_camera_bev_video.py
│   ├── 12_1_below_surface_clusters_video.py
│   ├── 12_2_mathworks_baseline_roadline.py
│   ├── 12_3_segformer_road_corridor_refined.py
│   ├── 13_inspect_kitti_object_dataset.py
│   ├── 14_1_check_image_label_alignment.py
│   ├── 14_2_project_lidar_to_object_image.py
│   ├── 14_3_debug_lidar_labels_on_image.py
│   ├── 14_4_draw_3d_boxes_on_image.py
│   ├── 14_5_lidar_points_inside_3d_boxes.py
│   ├── 14_6_draw_3d_boxes_on_bev.py
│   ├── 14_7_find_interesting_object_frames.py
│   ├── 14_8_generate_bev_gt_mask.py
│   ├── 15_1_generate_bev_training_dataset.py
│   ├── 15_2_inspect_bev_training_sample.py
│   ├── 15_3_train_bev_vehicle_segmentation.py
│   ├── 15_4_train_results.py
│   ├── 15_5_evaluate_small_unet.py
│   ├── 15_6_train_bev_vehicle_resnet.py
│   └── classes/
│       ├── KITTIBase.py                     # Data loading, calibration, projection
│       ├── KITTIDepthVisualizer.py          # Sparse depth maps & videos
│       ├── KITTIBEVBuilder.py               # BEV coordinate transforms & maps
│       ├── KITTIViewComposer.py             # Multi-panel video composer
│       ├── KITTIGroundEstimator.py          # RANSAC ground estimation
│       ├── KITTILidarClusterer.py           # LiDAR point clustering
│       ├── KITTILaneBaseline.py             # Baseline lane-line detection
│       ├── KITTISegFormerRoadSegmenter.py   # SegFormer road segmentation
│       ├── KITTIRoadFusion.py               # Camera + LiDAR road fusion
│       ├── KITTISemanticTools.py            # Semantic label utilities
│       ├── KITTIObjectBase.py               # KITTI object dataset reader
│       ├── KITTIObjectGeometry.py           # 3-D box geometry & transforms
│       ├── KITTIObjectVisualizer.py         # 3-D box visualisation
│       ├── KITTIBEVTrainingDatasetBuilder.py
│       ├── KITTIBEVSegmentationTrainer.py   # U-Net / ResNet training loop
│       ├── KITTIBEVSegmentationEvaluator.py
│       ├── KITTIBEVTrainingInspector.py
│       └── KITTIBEVTrainingResults.py
├── tests/
│   ├── test_kitti_base_projection.py   # Projection & filtering math
│   ├── test_bev_builder.py             # BEV coordinate & ROI logic
│   └── test_object_geometry.py        # 3-D box geometry
├── outputs/                            # Generated images, videos, checkpoints
├── models/                             # Optional pre-trained weights
├── requirements.txt
└── .gitignore
```
 
---

## Setup
 
**1. Clone and install dependencies**
 
```bash
git clone https://github.com/your-username/kitti_perception_pipeline.git
cd kitti_perception_pipeline
pip install -r requirements.txt
```
 
Recommended Python: **3.11**
 
**2. Download the KITTI datasets**
 
| Dataset | Link |
|---|---|
| KITTI Raw Data | https://www.cvlibs.net/datasets/kitti/raw_data.php |
| KITTI Object Detection | https://www.cvlibs.net/datasets/kitti/eval_object.php |
 
**3. Configure paths**
 
Copy the example env file and fill in your local paths:
 
```bash
cp config/.env.example config/.env
```
 
```env
KITTI_RAW_DIR=/path/to/kitti_raw
KITTI_OBJ_DIR=/path/to/kitti_object
KITTI_DATE=2011_09_26
KITTI_DRIVE=0019
OUTPUT_DIR=outputs
```
 
Verify with:
 
```bash
python config/config.py
```
 
---
 
## Usage
 
Scripts are self-contained and run in order from `scripts/`:
 
```bash
cd scripts
python 01_inspect_dataset.py
python 09_2_bev_occupancy_map.py
python 15_3_train_bev_vehicle_segmentation.py
```
 
All outputs (images, videos, model checkpoints) are saved to `outputs/`.
 
---
 
## Pipeline Stages
 
| Stage | Scripts | Description |
|---|---|---|
| Data Inspection | 01–02 | Dataset stats, calibration matrix verification |
| Projection | 03–05 | LiDAR → camera coords → image pixels |
| Depth Maps | 06–08 | Sparse depth images and overlay videos |
| BEV Maps | 09–10 | Occupancy / density / height / intensity |
| Ground Estimation | 11 | RANSAC plane fitting, above/below separation |
| Road & Lane | 12 | LiDAR clustering + SegFormer camera fusion |
| Object Dataset | 13–14 | 3-D box visualisation, GT mask generation |
| Deep Learning | 15 | BEV vehicle segmentation training & eval |
 
---

 ## Models
 
Two architectures are supported for BEV vehicle segmentation:
 
| Model | Input size | Script |
|---|---|---|
| Small U-Net | 448 × 320 | `15_3`, `15_5` |
| Small ResNet | 352 × 256 | `15_6` |
 
Training hyperparameters (epochs, batch size, learning rate, positive class weight) are configurable at the top of each training script.
 
---
 
## Running the Tests
 
Unit tests cover the pure-math functions (projection, BEV coordinate conversion, 3-D box geometry) and require no KITTI data files.
 
```bash
pip install pytest
pytest tests/ -v
```
 
---
 
## License
 
This project is for research and educational purposes only.  
The KITTI dataset is subject to its own [license terms](https://www.cvlibs.net/datasets/kitti/).

