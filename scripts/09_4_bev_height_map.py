import os
import numpy as np
import cv2
import pykitti


BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"


def filter_roi(points, x_min, x_max, y_min, y_max, z_min, z_max):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z < z_max)
    )

    return points[mask]


def metric_to_bev_pixels(points, x_min, x_max, y_min, y_max, resolution):
    x = points[:, 0]
    y = points[:, 1]

    rows = ((x_max - x) / resolution).astype(np.int32)
    cols = ((y_max - y) / resolution).astype(np.int32)

    bev_height = int((x_max - x_min) / resolution)
    bev_width = int((y_max - y_min) / resolution)

    valid = (
        (rows >= 0) & (rows < bev_height) &
        (cols >= 0) & (cols < bev_width)
    )

    return rows[valid], cols[valid], valid, bev_height, bev_width


def make_height_map(points, x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    points_roi = filter_roi(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max
    )

    rows, cols, valid, bev_height, bev_width = metric_to_bev_pixels(
        points_roi,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution
    )

    z_values = points_roi[:, 2][valid]

    # Start with -inf so any real z value will be larger.
    height_map = np.full((bev_height, bev_width), -np.inf, dtype=np.float32)

    for r, c, z in zip(rows, cols, z_values):
        if z > height_map[r, c]:
            height_map[r, c] = z

    empty = height_map == -np.inf

    # For visualization, set empty cells to z_min.
    height_for_vis = height_map.copy()
    height_for_vis[empty] = z_min

    # Normalize z range to 0-255.
    height_norm = (height_for_vis - z_min) / (z_max - z_min)
    height_norm = np.clip(height_norm, 0.0, 1.0)
    height_img = (height_norm * 255).astype(np.uint8)

    # Make empty cells black.
    height_img[empty] = 0

    return height_img, height_map, points_roi


def main():
    os.makedirs("../outputs/images", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE)
    velo = data.get_velo(0)

    points = velo[:, :3]

    x_min, x_max = 0.0, 50.0
    y_min, y_max = -25.0, 25.0
    z_min, z_max = -3.0, 2.0
    resolution = 0.10

    height_img, raw_height_map, points_roi = make_height_map(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        resolution
    )

    out_path_gray = "../outputs/images/bev_height_000000.png"
    cv2.imwrite(out_path_gray, height_img)

    # Optional: colorized height map
    color_height = cv2.applyColorMap(height_img, cv2.COLORMAP_JET)
    color_height[height_img == 0] = (0, 0, 0)

    out_path_color = "../outputs/images/bev_height_color_000000.png"
    cv2.imwrite(out_path_color, color_height)

    valid_cells = raw_height_map != -np.inf

    print("=== Phase 7.4: BEV Height Map ===")
    print("points original:", points.shape[0])
    print("points ROI:", points_roi.shape[0])
    print("height image shape:", height_img.shape)
    print("nonempty height cells:", np.count_nonzero(valid_cells))
    print("min height:", raw_height_map[valid_cells].min())
    print("max height:", raw_height_map[valid_cells].max())
    print("mean height:", raw_height_map[valid_cells].mean())
    print("saved:", out_path_gray)
    print("saved:", out_path_color)


if __name__ == "__main__":
    main()
