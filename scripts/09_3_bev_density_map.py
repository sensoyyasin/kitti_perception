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

    return rows[valid], cols[valid], bev_height, bev_width


def make_density_map(points, x_min, x_max, y_min, y_max, z_min, z_max, resolution, max_count=5):
    points_roi = filter_roi(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max
    )

    rows, cols, bev_height, bev_width = metric_to_bev_pixels(
        points_roi,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution
    )

    density = np.zeros((bev_height, bev_width), dtype=np.float32)

    for r, c in zip(rows, cols):
        density[r, c] += 1.0

    raw_density = density.copy()

    # Clip and normalize to 0-255 for visualization.
    density = np.clip(density, 0, max_count)
    density = (density / max_count * 255).astype(np.uint8)

    return density, raw_density, points_roi


def main():
    os.makedirs("../outputs/images", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE)
    velo = data.get_velo(0)

    points = velo[:, :3]

    x_min, x_max = 0.0, 50.0
    y_min, y_max = -25.0, 25.0
    z_min, z_max = -3.0, 2.0
    resolution = 0.10
    max_count = 20

    density_img, raw_density, points_roi = make_density_map(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        resolution,
        max_count=max_count
    )

    out_path = "../outputs/images/bev_density_000000.png"
    cv2.imwrite(out_path, density_img)

    print("=== Phase 7.3: BEV Density Map ===")
    print("points original:", points.shape[0])
    print("points ROI:", points_roi.shape[0])
    print("density image shape:", density_img.shape)
    print("nonzero cells:", np.count_nonzero(raw_density))
    print("max raw density in a cell:", raw_density.max())
    print("mean nonzero density:", raw_density[raw_density > 0].mean())
    print("saved:", out_path)


if __name__ == "__main__":
    main()
