from pathlib import Path
import numpy as np
import cv2


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"
FRAME_ID = "000613"

OUTPUT_DIR = Path("../outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# BEV region
X_MIN, X_MAX = 0.0, 70.0
Y_MIN, Y_MAX = -25.0, 25.0
Z_MIN, Z_MAX = -3.0, 2.0
RESOLUTION = 0.10


VEHICLE_CLASSES = {"Car", "Van", "Truck"}


CLASS_COLORS = {
    "Car": (0, 255, 0),
    "Van": (0, 200, 0),
    "Truck": (0, 160, 0),
    "Pedestrian": (255, 0, 0),
    "Person_sitting": (255, 120, 0),
    "Cyclist": (0, 255, 255),
    "Tram": (180, 0, 255),
    "Misc": (180, 180, 180),
    "DontCare": (80, 80, 80),
}


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
    """
    KITTI label box:
      dimensions = h, w, l
      location = bottom-center in camera coordinates
      rotation_y = yaw around camera Y axis
    """
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


def filter_roi(points_xyz):
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    mask = (
        (x >= X_MIN) & (x < X_MAX) &
        (y >= Y_MIN) & (y < Y_MAX) &
        (z >= Z_MIN) & (z < Z_MAX)
    )

    return points_xyz[mask], mask


def metric_to_bev(points_xyz):
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]

    height = int((X_MAX - X_MIN) / RESOLUTION)
    width = int((Y_MAX - Y_MIN) / RESOLUTION)

    rows = ((X_MAX - x) / RESOLUTION).astype(np.int32)
    cols = ((Y_MAX - y) / RESOLUTION).astype(np.int32)

    valid = (
        (rows >= 0) & (rows < height) &
        (cols >= 0) & (cols < width)
    )

    return rows[valid], cols[valid], valid, height, width


def bev_shape():
    height = int((X_MAX - X_MIN) / RESOLUTION)
    width = int((Y_MAX - Y_MIN) / RESOLUTION)
    return height, width


def make_bev_occupancy(points_velo):
    points_roi, _ = filter_roi(points_velo[:, :3])

    rows, cols, valid, h, w = metric_to_bev(points_roi)

    bev = np.zeros((h, w, 3), dtype=np.uint8)
    bev[rows, cols] = (220, 220, 220)

    return bev


def get_bottom_corners_bev_polygon(obj, calib):
    corners_cam = compute_3d_box_corners_camera(obj)
    corners_velo = transform_rect_camera_to_velodyne(corners_cam, calib)

    # Bottom face only: corners 0,1,2,3
    bottom = corners_velo[:4, :]

    rows, cols, valid, h, w = metric_to_bev(bottom)

    if len(rows) != 4:
        return None, corners_velo

    polygon = np.column_stack([cols, rows]).astype(np.int32)

    return polygon, corners_velo


def draw_bev_box(bev, polygon, color, thickness=2):
    cv2.polylines(
        bev,
        [polygon],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    center = np.mean(polygon, axis=0).astype(np.int32)
    front_mid = ((polygon[0] + polygon[1]) / 2).astype(np.int32)

    cv2.line(
        bev,
        tuple(center),
        tuple(front_mid),
        color,
        thickness,
        cv2.LINE_AA,
    )


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


def add_footer(image, text):
    out = image.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def main():
    root = BASEDIR / SPLIT

    velo_path = root / "velodyne" / f"{FRAME_ID}.bin"
    label_path = root / "label_2" / f"{FRAME_ID}.txt"
    calib_path = root / "calib" / f"{FRAME_ID}.txt"

    print("=== 14_8 Generate BEV GT Vehicle Mask ===")
    print("frame:", FRAME_ID)
    print("velodyne:", velo_path)
    print("label:", label_path)
    print("calib:", calib_path)

    points_velo = read_velodyne_bin(velo_path)
    objects = read_label_file(label_path)
    calib = read_calib_file(calib_path)

    h, w = bev_shape()

    bev = make_bev_occupancy(points_velo)
    mask = np.zeros((h, w), dtype=np.uint8)

    debug = bev.copy()

    vehicle_count = 0
    boxes_drawn = 0
    boxes_skipped = 0

    print()
    print("BEV shape:", mask.shape)
    print("ROI:")
    print("x:", X_MIN, X_MAX)
    print("y:", Y_MIN, Y_MAX)
    print("resolution:", RESOLUTION)
    print("objects:", len(objects))

    for idx, obj in enumerate(objects):
        cls = obj["type"]

        if cls not in VEHICLE_CLASSES:
            continue

        vehicle_count += 1

        polygon, corners_velo = get_bottom_corners_bev_polygon(obj, calib)

        center_velo = transform_rect_camera_to_velodyne(
            obj["location_xyz_camera"].reshape(1, 3),
            calib,
        )[0]

        print()
        print("vehicle object", idx)
        print("type:", cls)
        print("camera location:", obj["location_xyz_camera"])
        print("velodyne center approx:", center_velo)

        if polygon is None:
            boxes_skipped += 1
            print("mask polygon: skipped, outside BEV ROI")
            continue

        # Fill mask with value 1 for vehicles.
        cv2.fillPoly(mask, [polygon], 1)

        color = CLASS_COLORS.get(cls, (0, 255, 0))
        draw_bev_box(debug, polygon, color, thickness=2)

        boxes_drawn += 1

        print("mask polygon:", polygon.tolist())

    # Visualization mask
    mask_vis = (mask * 255).astype(np.uint8)
    mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # Make mask semi-visible with contour
    contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_vis_bgr, contours, -1, (0, 255, 0), 2)

    debug = add_title(debug, f"BEV GT Vehicle Boxes | Frame {FRAME_ID}")
    debug = add_footer(
        debug,
        f"gray=LiDAR ROI | boxes=vehicles | vehicle_objs={vehicle_count} drawn={boxes_drawn} skipped={boxes_skipped}",
    )

    mask_vis_bgr = add_title(mask_vis_bgr, "BEV Vehicle Mask")
    mask_vis_bgr = add_footer(mask_vis_bgr, "white/green = vehicle ground-truth mask")

    combined = np.hstack([debug, mask_vis_bgr])

    debug_out_path = OUTPUT_DIR / f"bev_vehicle_mask_debug_{FRAME_ID}.png"
    mask_out_path = OUTPUT_DIR / f"bev_vehicle_mask_{FRAME_ID}.png"

    cv2.imwrite(str(debug_out_path), combined)
    cv2.imwrite(str(mask_out_path), mask_vis)

    print()
    print("vehicle objects:", vehicle_count)
    print("boxes drawn:", boxes_drawn)
    print("boxes skipped:", boxes_skipped)
    print("mask positive pixels:", int(np.sum(mask > 0)))
    print("saved debug:", debug_out_path)
    print("saved mask:", mask_out_path)


if __name__ == "__main__":
    main()
