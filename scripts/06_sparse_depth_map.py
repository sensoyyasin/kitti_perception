import os
import numpy as np
import pykitti
import cv2


BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"


def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1))
    return np.hstack([points_xyz, ones])


def depth_to_bgr(depth, max_depth=80.0):
    # Depth value to BGR, = Near yellow/red, far = blue
    d = np.clip(depth, 0.0, max_depth)
    ratio = d / max_depth

    red = int(255 * (1.0 - ratio))
    green = int(255 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(255 * ratio)

    return (blue, green, red)


def main():
    os.makedirs("../outputs/images", exist_ok=True)
    os.makedirs("../outputs/npy", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    img_cam2, _ = data.get_rgb(0)
    width, height = img_cam2.size

    velo = data.get_velo(0)
    points_velo = velo[:, :3]

    # LiDAR -> camera
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

    # Camera -> image
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

    # Sparse depth map
    depth_map = np.zeros((height, width), dtype=np.float32)

    for px, py, d in zip(u, v, depth):
        current = depth_map[py, px]

        # If pixel is empty or new point is closer, keep new point.
        if current == 0 or d < current:
            depth_map[py, px] = d

    # Save raw depth map
    np.save("../outputs/npy/sparse_depth_000000.npy", depth_map)

    # Visualize sparse depth as image
    depth_vis = np.zeros((height, width, 3), dtype=np.uint8)

    nonzero = depth_map > 0
    ys, xs = np.where(nonzero)

    for py, px in zip(ys, xs):
        d = depth_map[py, px]
        depth_vis[py, px] = depth_to_bgr(d)

    cv2.imwrite("../outputs/images/sparse_depth_000000.png", depth_vis)

    print("=== Sparse Depth Map ===")
    print("image size:", width, height)
    print("projected points:", len(depth))
    print("nonzero depth pixels:", np.count_nonzero(depth_map))
    print("saved: ../outputs/npy/sparse_depth_000000.npy")
    print("saved: ../outputs/images/sparse_depth_000000.png")


if __name__ == "__main__":
    main()
