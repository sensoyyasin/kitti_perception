'''
velodyne bin -> BEV tensor
label + calib -> BEV vehicle mask
save .npy + .png

processed_bev/inputs/000000.npy			-> BEV feature tensor
processed_bev/masks_vehicle/000000.png	-> vehicle ground-truth mask
processed_bev/debug/000000.png			-> kontrol görseli

In every BEV : shape = (H, W, 4)

occupancy -> where points exist
density   -> how many points are there
height    -> vertical structure
intensity -> surface reflectance

0 -> occupancy
1 -> density
2 -> height
3 -> intensity

x: 0 - 70 m
y: -25 - 25 m
ROI:
resolution: 0.10 m

new Shape:
H = 700
W = 500
C = 4

It's better input for deep learning.
'''

from pathlib import Path
import numpy as np
import cv2


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"

INPUT_DIR = BASEDIR / "processed_bev" / "inputs"
MASK_DIR = BASEDIR / "processed_bev" / "masks_vehicle"
DEBUG_DIR = BASEDIR / "processed_bev" / "debug"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# We only extracted 1000 frames.
MAX_FRAMES = 1000


# BEV region
X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -25.0, 25.0
Z_MIN, Z_MAX = -3.0, 2.0
RESOLUTION = 0.10

VEHICLE_CLASSES = {"Car", "Van", "Truck"}


def read_velodyne_bin(path):
    points = np.fromfile(str(path), dtype=np.float32)
    return points.reshape(-1, 4)


def read_label_file(label_path):
    objects = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            obj = {
                "type": parts[0],
                "truncation": float(parts[1]),
                "occlusion": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": np.array([float(v) for v in parts[4:8]], dtype=np.float32),
                "dimensions_hwl": np.array([float(v) for v in parts[8:11]], dtype=np.float32),
                "location_xyz_camera": np.array([float(v) for v in parts[11:14]], dtype=np.float32),
                "rotation_y": float(parts[14]),
            }

            objects.append(obj)

    return objects


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


def get_transforms(calib):
    R0_rect = calib["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_rect_4x4 = make_4x4_from_3x3(R0_rect)
    Tr_velo_to_cam_4x4 = make_4x4_from_3x4(Tr_velo_to_cam)

    T_velo_to_rect_cam = R0_rect_4x4 @ Tr_velo_to_cam_4x4
    T_rect_cam_to_velo = np.linalg.inv(T_velo_to_rect_cam)

    return T_velo_to_rect_cam, T_rect_cam_to_velo


def transform_rect_camera_to_velodyne(points_cam, calib):
    _, T_rect_cam_to_velo = get_transforms(calib)

    points_cam = points_cam.astype(np.float64)
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    points_cam_h = np.hstack([points_cam, ones])

    points_velo_h = (T_rect_cam_to_velo @ points_cam_h.T).T

    return points_velo_h[:, :3]


def compute_3d_box_corners_camera(obj):
    h, w, l = obj["dimensions_hwl"]
    x, y, z = obj["location_xyz_camera"]
    ry = obj["rotation_y"]

    x_corners = np.array([
        l / 2,  l / 2, -l / 2, -l / 2,
        l / 2,  l / 2, -l / 2, -l / 2,
    ])

    y_corners = np.array([
        0, 0, 0, 0,
        -h, -h, -h, -h,
    ])

    z_corners = np.array([
        w / 2, -w / 2, -w / 2,  w / 2,
        w / 2, -w / 2, -w / 2,  w / 2,
    ])

    corners_obj = np.vstack([x_corners, y_corners, z_corners])

    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0,          1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    corners_cam = R @ corners_obj
    corners_cam[0, :] += x
    corners_cam[1, :] += y
    corners_cam[2, :] += z

    return corners_cam.T


def bev_shape():
    height = int((X_MAX - X_MIN) / RESOLUTION)
    width = int((Y_MAX - Y_MIN) / RESOLUTION)
    return height, width


def filter_roi(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= X_MIN) & (x < X_MAX) &
        (y >= Y_MIN) & (y < Y_MAX) &
        (z >= Z_MIN) & (z < Z_MAX)
    )

    return points[mask], mask


def metric_to_bev(points_xyz):
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]

    h, w = bev_shape()

    rows = ((X_MAX - x) / RESOLUTION).astype(np.int32)
    cols = ((Y_MAX - y) / RESOLUTION).astype(np.int32)

    valid = (
        (rows >= 0) & (rows < h) &
        (cols >= 0) & (cols < w)
    )

    return rows[valid], cols[valid], valid


def make_bev_tensor(points_velo):
    """
    Returns H x W x 4 float32 tensor:
      channel 0: occupancy
      channel 1: density
      channel 2: height
      channel 3: intensity
    """
    h, w = bev_shape()

    points_roi, _ = filter_roi(points_velo)

    xyz = points_roi[:, :3]
    reflectance = points_roi[:, 3]

    rows, cols, valid = metric_to_bev(xyz)

    xyz_valid = xyz[valid]
    reflectance_valid = reflectance[valid]

    occupancy = np.zeros((h, w), dtype=np.float32)
    density = np.zeros((h, w), dtype=np.float32)
    height_map = np.full((h, w), -np.inf, dtype=np.float32)
    intensity = np.zeros((h, w), dtype=np.float32)

    occupancy[rows, cols] = 1.0

    np.add.at(density, (rows, cols), 1.0)

    z_values = xyz_valid[:, 2]
    np.maximum.at(height_map, (rows, cols), z_values)

    np.maximum.at(intensity, (rows, cols), reflectance_valid)

    # Normalize density with log scaling.
    density = np.log1p(density) / np.log(16.0)
    density = np.clip(density, 0.0, 1.0)

    # Normalize height.
    empty = height_map == -np.inf
    height_map[empty] = Z_MIN
    height_map = (height_map - Z_MIN) / (Z_MAX - Z_MIN)
    height_map = np.clip(height_map, 0.0, 1.0)
    height_map[empty] = 0.0

    intensity = np.clip(intensity, 0.0, 1.0)

    bev_tensor = np.stack(
        [occupancy, density, height_map, intensity],
        axis=-1,
    ).astype(np.float32)

    return bev_tensor


def get_bottom_corners_bev_polygon(obj, calib):
    corners_cam = compute_3d_box_corners_camera(obj)
    corners_velo = transform_rect_camera_to_velodyne(corners_cam, calib)

    bottom = corners_velo[:4, :]

    rows, cols, valid = metric_to_bev(bottom)

    if len(rows) != 4:
        return None

    polygon = np.column_stack([cols, rows]).astype(np.int32)

    return polygon


def make_vehicle_mask(objects, calib):
    h, w = bev_shape()

    mask = np.zeros((h, w), dtype=np.uint8)

    vehicle_count = 0
    drawn_count = 0
    skipped_count = 0

    for obj in objects:
        cls = obj["type"]

        if cls not in VEHICLE_CLASSES:
            continue

        vehicle_count += 1

        polygon = get_bottom_corners_bev_polygon(obj, calib)

        if polygon is None:
            skipped_count += 1
            continue

        cv2.fillPoly(mask, [polygon], 1)
        drawn_count += 1

    stats = {
        "vehicle_count": vehicle_count,
        "drawn_count": drawn_count,
        "skipped_count": skipped_count,
        "mask_pixels": int(np.sum(mask > 0)),
    }

    return mask, stats


def make_debug_image(bev_tensor, mask, frame_id, stats):
    occupancy = (bev_tensor[:, :, 0] * 255).astype(np.uint8)
    density = (bev_tensor[:, :, 1] * 255).astype(np.uint8)
    height = (bev_tensor[:, :, 2] * 255).astype(np.uint8)
    intensity = (bev_tensor[:, :, 3] * 255).astype(np.uint8)

    occ_bgr = cv2.cvtColor(occupancy, cv2.COLOR_GRAY2BGR)
    density_bgr = cv2.cvtColor(density, cv2.COLOR_GRAY2BGR)
    height_bgr = cv2.applyColorMap(height, cv2.COLORMAP_JET)
    height_bgr[height == 0] = (0, 0, 0)
    intensity_bgr = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    intensity_bgr[intensity == 0] = (0, 0, 0)

    mask_vis = (mask * 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_bgr, contours, -1, (0, 255, 0), 2)

    def title(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            out,
            text,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    occ_bgr = title(occ_bgr, "Occupancy")
    density_bgr = title(density_bgr, "Density")
    height_bgr = title(height_bgr, "Height")
    intensity_bgr = title(intensity_bgr, "Intensity")
    mask_bgr = title(mask_bgr, "Vehicle Mask")

    top = np.hstack([occ_bgr, density_bgr])
    bottom = np.hstack([height_bgr, intensity_bgr])

    features = np.vstack([top, bottom])

    mask_bgr = cv2.resize(
        mask_bgr,
        (features.shape[1], features.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    combined = np.hstack([features, mask_bgr])

    footer = (
        f"Frame {frame_id} | vehicles={stats['vehicle_count']} "
        f"drawn={stats['drawn_count']} skipped={stats['skipped_count']} "
        f"mask_pixels={stats['mask_pixels']}"
    )

    cv2.rectangle(
        combined,
        (0, combined.shape[0] - 36),
        (combined.shape[1], combined.shape[0]),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        combined,
        footer,
        (12, combined.shape[0] - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return combined


def main():
    root = BASEDIR / SPLIT

    image_dir = root / "image_2"
    velo_dir = root / "velodyne"
    label_dir = root / "label_2"
    calib_dir = root / "calib"

    frame_ids = sorted(p.stem for p in velo_dir.glob("*.bin"))
    frame_ids = frame_ids[:MAX_FRAMES]

    print("=== 14_9 Generate BEV Training Dataset ===")
    print("frames:", len(frame_ids))
    print("output inputs:", INPUT_DIR)
    print("output masks:", MASK_DIR)
    print("output debug:", DEBUG_DIR)
    print("BEV shape:", bev_shape())
    print()

    total_vehicle_count = 0
    total_drawn_count = 0
    total_skipped_count = 0

    for idx, frame_id in enumerate(frame_ids):
        velo_path = velo_dir / f"{frame_id}.bin"
        label_path = label_dir / f"{frame_id}.txt"
        calib_path = calib_dir / f"{frame_id}.txt"

        if not label_path.exists() or not calib_path.exists():
            print("missing label/calib for", frame_id)
            continue

        points_velo = read_velodyne_bin(velo_path)
        objects = read_label_file(label_path)
        calib = read_calib_file(calib_path)

        bev_tensor = make_bev_tensor(points_velo)
        mask, stats = make_vehicle_mask(objects, calib)

        np.save(INPUT_DIR / f"{frame_id}.npy", bev_tensor)
        cv2.imwrite(str(MASK_DIR / f"{frame_id}.png"), (mask * 255).astype(np.uint8))

        total_vehicle_count += stats["vehicle_count"]
        total_drawn_count += stats["drawn_count"]
        total_skipped_count += stats["skipped_count"]

        # Save debug for first 10 and every 100 frames.
        if idx < 10 or idx % 100 == 0:
            debug = make_debug_image(bev_tensor, mask, frame_id, stats)
            cv2.imwrite(str(DEBUG_DIR / f"{frame_id}.png"), debug)

        if idx % 50 == 0:
            print(
                f"processed {idx}/{len(frame_ids)} frame={frame_id} "
                f"vehicles={stats['vehicle_count']} "
                f"drawn={stats['drawn_count']} "
                f"skipped={stats['skipped_count']} "
                f"mask_pixels={stats['mask_pixels']}"
            )

    print()
    print("done")
    print("total vehicle objects:", total_vehicle_count)
    print("total drawn boxes:", total_drawn_count)
    print("total skipped boxes:", total_skipped_count)
    print("saved inputs:", INPUT_DIR)
    print("saved masks:", MASK_DIR)
    print("saved debug:", DEBUG_DIR)


if __name__ == "__main__":
    main()
