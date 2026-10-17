# KITTI Perception Pipeline

An end-to-end LiDAR-camera fusion and Bird's Eye-View (BEV) vehicle segmentation pipeline built on the KITTI autonomous driving dataset, , with a Monte Carlo study of how fog degrades LiDAR perception.

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
- Monte Carlo robustness study: physically based fog (Beer–Lambert, α values from Hahner et al., ICCV 2021) applied to real LiDAR scans, measuring vehicle visibility by distance


In this project, I used the KITTI raw data to build a step-by-step geometric perception pipeline. I inspected the dataset structure and calibration files, worked with the intrinsic and extrinsic matrices, transformed LiDAR points from the Velodyne frame into the camera frame, and projected them onto the cam2 RGB image. After validating the projection, I generated sparse depth maps and camera-LiDAR depth overlays. I also converted the LiDAR point cloud into Bird's Eye View (BEV) maps, including occupancy, density, height, and intensity representations. BEV is useful because it converts the 3D scene into a top-down metric grid, making distances, road layout, point density, and object positions easier to reason about. I also tested ground plane estimation using RANSAC to separate road-like ground points from above-ground and below-ground residual points. The goal of this project is not to train a black-box model, but to understand the core geometry behind camera-LiDAR sensor fusion and build a clean perception front-end from raw autonomous driving sensor data. On top of this front-end, I asked how robust the LiDAR data itself is: by simulating fog thousands of times over real scans, I found that its effective range in fog is much shorter than human visibility.

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
│   ├── 16_monte_carlo_fog.py
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
| Robustness | 16 | Monte Carlo study: LiDAR point loss in fog |
 
---
 
## Models
 
Two architectures are supported for BEV vehicle segmentation:
 
| Model | Input size | Script |
|---|---|---|
| Small U-Net | 448 × 320 | `15_3`, `15_5` |
| Small ResNet | 352 × 256 | `15_6` |
 
Training hyperparameters (epochs, batch size, learning rate, positive class weight) are configurable at the top of each training script.
 
---
 
## Robustness Study: How Much Can LiDAR See in Fog? (Monte Carlo)

<img width="700" height="271" alt="monte_carlo" src="https://github.com/user-attachments/assets/37d5ae28-47b2-44d9-95c1-e6c9686b7eeb" />

 
**Question.** If a KITTI scene had been recorded in fog, how many LiDAR points
would each car still produce, and how many cars would effectively disappear?
 
**Method** (`scripts/16_monte_carlo_fog.py`, single self-contained file)
 
1. Count the LiDAR points inside each labelled vehicle box (Car, Van, Truck).
2. Apply fog: a LiDAR pulse crosses the fog twice (out and back), so by the
   Beer–Lambert law a point at distance *r* survives with probability
   `p = exp(-2 · α · r)`. One random draw per point decides whether it survives.
3. Repeat over 254 frames × 3 fog levels × 5 random seeds, and count how many
   vehicles still have at least 10 points ("visible").
**Fog levels.** α values are taken from Hahner et al., *Fog Simulation on Real
LiDAR Point Clouds for 3D Object Detection in Adverse Weather*, ICCV 2021.
Visibility (meteorological optical range) ≈ 3 / α:
 
| α (1/m) | Visibility | Condition |
|---|---|---|
| 0.01 | ~300 m | light fog |
| 0.03 | ~100 m | fog |
| 0.06 | ~50 m | dense fog |
 
**Results.** Share of vehicles that keep ≥ 10 LiDAR points:
 
| Fog α | 0–20 m | 20–40 m | 40–70 m |
|---|---|---|---|
| 0.01 | 100% | 96% | 52% |
| 0.03 | 100% | 69% | 4% |
| 0.06 | 98% | 24% | 0% |
 
**Findings**
 
- Fog removes far points first: close cars (0–20 m) stay visible even in dense
  fog, while cars beyond 40 m are lost in moderate fog.
- **LiDAR's effective range in fog is much shorter than human visibility.** In
  dense fog (~50 m visibility) only 24% of cars at 20–40 m keep enough points,
  because the pulse crosses the fog twice and a car needs many returns, not one.
- Simulated counts match the formula: in frame 000008 at α = 0.01 the car at
  ~34 m keeps 26 of its 53 points (49%), and `exp(-2 · 0.01 · 34) ≈ 0.51`.
**Limitations**
 
- Sensor level only: we count points, we do not run a detector.
- Only point loss is modelled. Hahner et al. also simulate fake points from fog
  backscatter and intensity loss.
- The 10-point "visible" threshold is a choice; results depend on it.
- 254 KITTI frames from one dataset.
**Run**
 
The script reads the KITTI 3D Object data (`velodyne`, `label_2`, `calib`) from
`~/datasets/kitti_object/training`. If your data is somewhere else, change `DATA_DIR`
at the top of `scripts/16_monte_carlo_fog.py`.
 
```bash
cd scripts
python 16_monte_carlo_fog.py   # prints the table, then opens two OpenCV windows
```
 
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
 
