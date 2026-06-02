import os
import numpy as np
import cv2
import pykitti


BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"
NUM_FRAMES = 481

def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.hstack([points_xyz, ones])


def depth_to_bgr(depth, max_depth=80.0):
    depth = np.clip(depth, 0.0, max_depth)
    ratio = depth / max_depth

    red = int(255 * (1.0 - ratio))
    green = int(255 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(255 * ratio)

    return (blue, green, red)


def project_lidar_to_camera(velo, K, T_cam2_velo, width, height):
    points_velo = velo[:, :3]

    points_velo_h = to_homogeneous(points_velo)
    points_cam_h = (T_cam2_velo @ points_velo_h.T).T
    points_cam = points_cam_h[:, :3]

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    in_front = Z > 0

    X = X[in_front]
    Y = Y[in_front]
    Z = Z[in_front]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    inside = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    u = u[inside].astype(np.int32)
    v = v[inside].astype(np.int32)
    depth = Z[inside]

    return u, v, depth


def draw_camera_overlay(img_cam2, velo, K, T_cam2_velo):
    img_rgb = np.array(img_cam2)
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    height, width = frame.shape[:2]

    u, v, depth = project_lidar_to_camera(
        velo,
        K,
        T_cam2_velo,
        width,
        height,
    )

    order = np.argsort(depth)[::-1]

    for idx in order:
        px = u[idx]
        py = v[idx]
        d = depth[idx]
        color = depth_to_bgr(d)
        cv2.circle(frame, (px, py), 1, color, -1)

    cv2.putText(
        frame,
        f"Camera + LiDAR Depth Overlay | points: {len(depth)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame, len(depth)


# -----------------------------
# BEV feature utils
# -----------------------------

def filter_roi(velo, x_min, x_max, y_min, y_max, z_min, z_max):
    x = velo[:, 0]
    y = velo[:, 1]
    z = velo[:, 2]

    mask = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z < z_max)
    )

    return velo[mask]


def metric_to_bev_pixels(points, x_min, x_max, y_min, y_max, resolution):
    """
    KITTI Velodyne coordinates:
        x = forward
        y = left
        z = up

    BEV image convention used here:
        row smaller = farther forward
        col smaller = vehicle left
        col larger  = vehicle right

    This makes BEV left/right visually match camera left/right more intuitively.
    """
    x = points[:, 0]
    y = points[:, 1]

    rows = ((x_max - x) / resolution).astype(np.int32)

    # Important orientation fix:
    # KITTI y positive means vehicle-left, so map larger y to smaller column.
    cols = ((y_max - y) / resolution).astype(np.int32)

    bev_height = int((x_max - x_min) / resolution)
    bev_width = int((y_max - y_min) / resolution)

    valid = (
        (rows >= 0) & (rows < bev_height) &
        (cols >= 0) & (cols < bev_width)
    )

    return rows[valid], cols[valid], valid, bev_height, bev_width


def make_bev_maps(velo, cfg):
    x_min, x_max, y_min, y_max, z_min, z_max, resolution = cfg

    velo_roi = filter_roi(
        velo,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )

    if velo_roi.shape[0] == 0:
        bev_height = int((x_max - x_min) / resolution)
        bev_width = int((y_max - y_min) / resolution)

        empty = np.zeros((bev_height, bev_width), dtype=np.uint8)
        stats = {
            "roi_points": 0,
            "occupied_cells": 0,
            "max_density": 0.0,
            "mean_height": 0.0,
        }
        return empty, empty, empty, empty, stats

    points_roi = velo_roi[:, :3]
    reflectance_roi = velo_roi[:, 3]

    rows, cols, valid, h, w = metric_to_bev_pixels(
        points_roi,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution,
    )

    if rows.shape[0] == 0:
        empty = np.zeros((h, w), dtype=np.uint8)
        stats = {
            "roi_points": int(velo_roi.shape[0]),
            "occupied_cells": 0,
            "max_density": 0.0,
            "mean_height": 0.0,
        }
        return empty, empty, empty, empty, stats

    z_values = points_roi[:, 2][valid]
    reflectance = reflectance_roi[valid]

    # Occupancy map
    occupancy = np.zeros((h, w), dtype=np.uint8)
    occupancy[rows, cols] = 255

    # Density map
    density = np.zeros((h, w), dtype=np.float32)
    for r, c in zip(rows, cols):
        density[r, c] += 1.0

    density_max_count = 5.0
    density_img = np.clip(density, 0.0, density_max_count)
    density_img = (density_img / density_max_count * 255.0).astype(np.uint8)

    # Height map: max z per BEV cell
    height_map = np.full((h, w), -np.inf, dtype=np.float32)

    for r, c, z in zip(rows, cols, z_values):
        if z > height_map[r, c]:
            height_map[r, c] = z

    valid_height_cells = height_map != -np.inf

    height_vis = height_map.copy()
    height_vis[~valid_height_cells] = z_min

    height_img = (height_vis - z_min) / (z_max - z_min)
    height_img = np.clip(height_img, 0.0, 1.0)
    height_img = (height_img * 255.0).astype(np.uint8)
    height_img[~valid_height_cells] = 0

    # Intensity map: max reflectance per BEV cell
    intensity_map = np.zeros((h, w), dtype=np.float32)

    for r, c, refl in zip(rows, cols, reflectance):
        if refl > intensity_map[r, c]:
            intensity_map[r, c] = refl

    intensity_img = np.clip(intensity_map, 0.0, 1.0)
    intensity_img = (intensity_img * 255.0).astype(np.uint8)

    stats = {
        "roi_points": int(velo_roi.shape[0]),
        "occupied_cells": int(np.count_nonzero(occupancy)),
        "max_density": float(density.max()) if density.size > 0 else 0.0,
        "mean_height": float(height_map[valid_height_cells].mean()) if np.any(valid_height_cells) else 0.0,
    }

    return occupancy, density_img, height_img, intensity_img, stats


# -----------------------------
# Visualization utils
# -----------------------------

def make_panel(gray, title, size, colormap=None):
    if colormap is None:
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        panel = cv2.applyColorMap(gray, colormap)
        panel[gray == 0] = (0, 0, 0)

    panel = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)

    cv2.putText(
        panel,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return panel


def make_stats_panel(size, frame_idx, projected_points, stats, fps):
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    lines = [
        "KITTI Pipeline",
        f"Frame: {frame_idx}",
        f"FPS: {fps:.2f}",
        f"Projected pts: {projected_points}",
        f"ROI pts: {stats['roi_points']}",
        f"Occupied cells: {stats['occupied_cells']}",
        f"Max density: {stats['max_density']:.1f}",
        f"Mean height: {stats['mean_height']:.2f} m",
        "",
        "Top:",
        "Camera + LiDAR depth",
        "Bottom:",
        "BEV feature maps",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return panel


def main():
    os.makedirs("../outputs/videos", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(NUM_FRAMES))

    duration = (data.timestamps[-1] - data.timestamps[0]).total_seconds()
    fps = (len(data.timestamps) - 1) / duration

    print("Using FPS:", fps)

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    # BEV config
    x_min, x_max = 0.0, 50.0
    y_min, y_max = -25.0, 25.0
    z_min, z_max = -3.0, 2.0
    resolution = 0.10
    cfg = (x_min, x_max, y_min, y_max, z_min, z_max, resolution)

    output_path = "../outputs/videos/all_views_single_video_fixed.mp4"

    # Fixed output layout.
    output_width = 1600
    camera_height = 480
    bottom_height = 320
    total_height = camera_height + bottom_height

    num_bottom_panels = 5
    panel_width = output_width // num_bottom_panels
    panel_size = (panel_width, bottom_height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, total_height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try a different codec or output path.")

    for i in range(NUM_FRAMES):
        img_cam2, _ = data.get_rgb(i)
        velo = data.get_velo(i)

        camera_panel, projected_points = draw_camera_overlay(
            img_cam2,
            velo,
            K,
            T_cam2_velo,
        )

        camera_panel = cv2.resize(
            camera_panel,
            (output_width, camera_height),
            interpolation=cv2.INTER_AREA,
        )

        occupancy, density, height, intensity, stats = make_bev_maps(velo, cfg)

        occ_panel = make_panel(
            occupancy,
            "Occupancy",
            panel_size,
            colormap=None,
        )

        density_panel = make_panel(
            density,
            "Density",
            panel_size,
            colormap=None,
        )

        height_panel = make_panel(
            height,
            "Height",
            panel_size,
            colormap=cv2.COLORMAP_JET,
        )

        intensity_panel = make_panel(
            intensity,
            "Intensity",
            panel_size,
            colormap=cv2.COLORMAP_TURBO,
        )

        stats_panel = make_stats_panel(
            panel_size,
            i,
            projected_points,
            stats,
            fps,
        )

        bottom = np.hstack([
            occ_panel,
            density_panel,
            height_panel,
            intensity_panel,
            stats_panel,
        ])

        # Safety: force exact size, because VideoWriter needs constant frame size.
        bottom = cv2.resize(
            bottom,
            (output_width, bottom_height),
            interpolation=cv2.INTER_AREA,
        )

        combined = np.vstack([camera_panel, bottom])

        # Safety check.
        if combined.shape[1] != output_width or combined.shape[0] != total_height:
            raise RuntimeError(
                f"Invalid frame shape: {combined.shape}, expected ({total_height}, {output_width}, 3)"
            )

        writer.write(combined)

        if i % 25 == 0:
            print(f"processed frame {i}/{NUM_FRAMES}")

    writer.release()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
