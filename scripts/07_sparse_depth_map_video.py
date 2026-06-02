import os
import numpy as np
import pykitti
import cv2


BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"
NUM_FRAMES = 481


def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1))
    return np.hstack([points_xyz, ones])


def depth_to_bgr_array(depth_map, max_depth=80.0):
    h, w = depth_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    valid = depth_map > 0
    d = np.clip(depth_map[valid], 0.0, max_depth)
    ratio = d / max_depth

    red = (255 * (1.0 - ratio)).astype(np.uint8)
    green = (255 * (1.0 - np.abs(ratio - 0.5) * 2.0)).astype(np.uint8)
    blue = (255 * ratio).astype(np.uint8)

    vis[valid, 0] = blue
    vis[valid, 1] = green
    vis[valid, 2] = red

    return vis


def make_sparse_depth_frame(velo, K, T_cam2_velo, width, height):
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

    depth_map = np.zeros((height, width), dtype=np.float32)

    for px, py, d in zip(u, v, depth):
        current = depth_map[py, px]

        if current == 0 or d < current:
            depth_map[py, px] = d

    depth_vis = depth_to_bgr_array(depth_map)
    return depth_vis, np.count_nonzero(depth_map), len(depth)


def main():
    os.makedirs("../outputs/videos", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(NUM_FRAMES))

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    first_img, _ = data.get_rgb(0)
    width, height = first_img.size

    output_path = "../outputs/videos/sparse_depth_video.mp4"
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(NUM_FRAMES):
        velo = data.get_velo(i)

        frame, nonzero_pixels, projected_points = make_sparse_depth_frame(
            velo,
            K,
            T_cam2_velo,
            width,
            height
        )

        cv2.putText(
            frame,
            f"Sparse Depth Map | Frame {i} | pixels: {nonzero_pixels}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        if i % 25 == 0:
            print(
                f"processed frame {i}/{NUM_FRAMES}, "
                f"projected={projected_points}, nonzero={nonzero_pixels}"
            )

    writer.release()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
