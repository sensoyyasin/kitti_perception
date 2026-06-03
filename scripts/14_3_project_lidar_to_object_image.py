from pathlib import Path
import numpy as np
import cv2


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"
FRAME_ID = "000000"

OUTPUT_DIR = Path("../outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_velodyne_bin(path):
    """
    KITTI velodyne format:
    [x, y, z, reflectance] as float32
    """
    points = np.fromfile(str(path), dtype=np.float32)
    points = points.reshape(-1, 4)
    return points


def read_calib_file(calib_path):
    calib = {}

    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            values = np.array([float(x) for x in value.strip().split()], dtype=np.float64)
            calib[key] = values

    return calib


def make_4x4_from_3x4(mat_3x4):
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = mat_3x4
    return out


def make_4x4_from_3x3(mat_3x3):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = mat_3x3
    return out


def depth_to_bgr(depth, max_depth=80.0):
    d = np.clip(depth, 0.0, max_depth)
    ratio = d / max_depth

    red = int(255 * (1.0 - ratio))
    green = int(255 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(255 * ratio)

    return (blue, green, red)


def project_velodyne_to_image(points_velo, calib, image_width, image_height):
    P2 = calib["P2"].reshape(3, 4)
    R0_rect = calib["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_rect_4x4 = make_4x4_from_3x3(R0_rect)
    Tr_velo_to_cam_4x4 = make_4x4_from_3x4(Tr_velo_to_cam)

    xyz = points_velo[:, :3]

    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.hstack([xyz.astype(np.float64), ones])

    points_cam_rect_h = (R0_rect_4x4 @ Tr_velo_to_cam_4x4 @ xyz_h.T).T

    depth = points_cam_rect_h[:, 2]
    in_front = depth > 0.1

    points_cam_rect_h = points_cam_rect_h[in_front]
    depth = depth[in_front]

    pixels_h = (P2 @ points_cam_rect_h.T).T

    u = pixels_h[:, 0] / pixels_h[:, 2]
    v = pixels_h[:, 1] / pixels_h[:, 2]

    inside = (
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )

    u = u[inside].astype(np.int32)
    v = v[inside].astype(np.int32)
    depth = depth[inside]

    return u, v, depth


def draw_lidar_overlay(image_bgr, u, v, depth):
    out = image_bgr.copy()

    # Draw far points first, near points last
    order = np.argsort(depth)[::-1]

    for idx in order:
        px = u[idx]
        py = v[idx]
        d = depth[idx]

        color = depth_to_bgr(d)
        cv2.circle(out, (px, py), 1, color, -1)

    return out


def add_title(image, title):
    out = image.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def main():
    root = BASEDIR / SPLIT

    image_path = root / "image_2" / f"{FRAME_ID}.png"
    velo_path = root / "velodyne" / f"{FRAME_ID}.bin"
    calib_path = root / "calib" / f"{FRAME_ID}.txt"

    print("=== 14_3 Project LiDAR to KITTI Object Image ===")
    print("frame:", FRAME_ID)
    print("image:", image_path)
    print("velodyne:", velo_path)
    print("calib:", calib_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points = read_velodyne_bin(velo_path)
    calib = read_calib_file(calib_path)

    height, width = image.shape[:2]

    print()
    print("image shape:", image.shape)
    print("velodyne shape:", points.shape)
    print("velodyne columns: x, y, z, reflectance")
    print("first 5 points:")
    print(points[:5])

    u, v, depth = project_velodyne_to_image(
        points,
        calib,
        width,
        height,
    )

    print()
    print("projected points inside image:", len(depth))
    print("depth min:", float(depth.min()) if len(depth) > 0 else None)
    print("depth max:", float(depth.max()) if len(depth) > 0 else None)
    print("depth mean:", float(depth.mean()) if len(depth) > 0 else None)

    raw = add_title(image, f"Raw image_2/{FRAME_ID}.png")
    overlay = draw_lidar_overlay(image, u, v, depth)
    overlay = add_title(overlay, f"LiDAR projected with calib/{FRAME_ID}.txt")

    combined = np.hstack([raw, overlay])

    out_path = OUTPUT_DIR / f"object_lidar_projection_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), combined)

    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
