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


def depth_to_bgr(depth, max_depth=80.0):
    d = np.clip(depth, 0.0, max_depth)
    ratio = d / max_depth

    red = int(255 * (1.0 - ratio))
    green = int(255 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(255 * ratio)

    return (blue, green, red)


def project_lidar(velo, K, T_cam2_velo, width, height):
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


def main():
    os.makedirs("../outputs/videos", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(NUM_FRAMES))

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    first_img, _ = data.get_rgb(0)
    width, height = first_img.size

    output_path = "../outputs/videos/camera_depth_overlay_video.mp4"
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(NUM_FRAMES):
        img_cam2, _ = data.get_rgb(i)
        velo = data.get_velo(i)

        img_rgb = np.array(img_cam2)
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        u, v, depth = project_lidar(
            velo,
            K,
            T_cam2_velo,
            width,
            height
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
            f"Camera + LiDAR Depth Overlay | Frame {i} | points: {len(depth)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        if i % 25 == 0:
            print(f"processed frame {i}/{NUM_FRAMES}, projected points={len(depth)}")

    writer.release()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
