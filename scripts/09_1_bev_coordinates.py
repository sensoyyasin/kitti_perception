import numpy as np
import pykitti

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"

def filter_roi(points, x_min, x_max, y_min, y_max, z_min, z_max):
    # Keep only points inside the region of interest.
    # KITTI Velodyne coordinates:
    # x = forward, y = left, z = up

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= x_min) & (x <= x_max) &
        (y >= y_min) & (y <= y_max) &
        (z >= z_min) & (z <= z_max)
    )

    return points[mask], mask


def metric_to_bev_pixels(points, x_min, x_max, y_min, y_max, resolution):
    # convert metric lidar coordinates to BEV image pixel coordinates.
    # In BEV image, row increases downward, col increases to the right.

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


def main():
    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    velo = data.get_velo(0)
    points = velo[:, :3]

    print("=== Phase 7.1: BEV coordinate conversion ===")
    print("Original point cloud shape:", points.shape)

    # Region of interest - roi
    x_min, x_max = 0.0, 50.0
    y_min, y_max = -25.0, 25.0
    z_min, z_max = -3.0, 2.0
    resolution = 0.10

    print("\nROI parameters:")
    print(f"x_min={x_min}, x_max={x_max}")
    print(f"y_min={y_min}, y_max={y_max}")
    print(f"z_min={z_min}, z_max={z_max}")
    print(f"resolution={resolution} meters/pixel")

    points_roi, roi_mask = filter_roi(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max
    )

    print("\nAfter ROI filtering:")
    print("points_roi shape:", points_roi.shape)
    print("points removed:", points.shape[0] - points_roi.shape[0])

    rows, cols, valid, bev_height, bev_width = metric_to_bev_pixels(
        points_roi,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution
    )

    print("\nBEV image size:")
    print("bev_height:", bev_height)
    print("bev_width:", bev_width)

    print("\nBEV pixel arrays:")
    print("rows shape:", rows.shape)
    print("cols shape:", cols.shape)

    print("\nRow stats:")
    print("min row:", rows.min())
    print("max row:", rows.max())

    print("\nCol stats:")
    print("min col:", cols.min())
    print("max col:", cols.max())

    print("\nFirst 10 ROI points and BEV pixels:")
    for i in range(min(10, len(rows))):
        x, y, z = points_roi[valid][i]
        print(
            f"point xyz=({x:.2f}, {y:.2f}, {z:.2f}) "
            f"-> row={rows[i]}, col={cols[i]}"
        )

    print("\nInterpretation:")
    print("x forward becomes BEV row.")
    print("y left/right becomes BEV column.")
    print("y=0 should be near the center column.")


if __name__ == "__main__":
    main()
