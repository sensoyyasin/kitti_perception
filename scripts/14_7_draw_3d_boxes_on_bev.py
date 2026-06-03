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
RESOLUTION = 0.05


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


def transform_velodyne_to_rect_camera(points_velo, calib):
    T_velo_to_rect_cam, _ = get_transforms(calib)

    xyz = points_velo[:, :3].astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.hstack([xyz, ones])

    points_cam_h = (T_velo_to_rect_cam @ xyz_h.T).T

    return points_cam_h[:, :3]


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


def points_inside_3d_box_camera(points_cam, obj):
    h, w, l = obj["dimensions_hwl"]
    center = obj["location_xyz_camera"].astype(np.float64)
    ry = obj["rotation_y"]

    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0,          1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    points_local = (R.T @ (points_cam - center).T).T

    x = points_local[:, 0]
    y = points_local[:, 1]
    z = points_local[:, 2]

    inside = (
        (x >= -l / 2) & (x <= l / 2) &
        (y >= -h) & (y <= 0) &
        (z >= -w / 2) & (z <= w / 2)
    )

    return inside


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

    height = int((X_MAX - X_MIN) / RESOLUTION)
    width = int((Y_MAX - Y_MIN) / RESOLUTION)

    rows = ((X_MAX - x) / RESOLUTION).astype(np.int32)
    cols = ((Y_MAX - y) / RESOLUTION).astype(np.int32)

    valid = (
        (rows >= 0) & (rows < height) &
        (cols >= 0) & (cols < width)
    )

    return rows[valid], cols[valid], valid, height, width


def make_bev_occupancy(points_velo):
    points_roi, roi_mask = filter_roi(points_velo[:, :3])

    rows, cols, valid, h, w = metric_to_bev(points_roi)

    bev = np.zeros((h, w, 3), dtype=np.uint8)
    bev[rows, cols] = (220, 220, 220)

    return bev, roi_mask


def draw_bev_points(bev, points_xyz, color, radius=1):
    rows, cols, valid, h, w = metric_to_bev(points_xyz)

    for r, c in zip(rows, cols):
        cv2.circle(bev, (int(c), int(r)), radius, color, -1)


def draw_bev_box(bev, corners_velo, color, thickness=2):
    """
    Draw bottom face of 3D box on BEV.
    corners 0-3 are bottom corners in camera convention after conversion.
    """
    bottom = corners_velo[:4, :]

    rows, cols, valid, h, w = metric_to_bev(bottom)

    if len(rows) != 4:
        return False

    pts = np.column_stack([cols, rows]).astype(np.int32)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for i, j in edges:
        cv2.line(
            bev,
            tuple(pts[i]),
            tuple(pts[j]),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Draw a small direction line from center to front edge midpoint.
    center = np.mean(pts, axis=0).astype(np.int32)
    front_mid = ((pts[0] + pts[1]) / 2).astype(np.int32)
    cv2.line(bev, tuple(center), tuple(front_mid), color, thickness, cv2.LINE_AA)

    return True


def add_title(bev, title):
    out = bev.copy()
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


def add_footer(bev, text):
    out = bev.copy()
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

    print("=== 14_7 Draw 3D Boxes on BEV ===")
    print("frame:", FRAME_ID)
    print("velodyne:", velo_path)
    print("label:", label_path)
    print("calib:", calib_path)

    points_velo = read_velodyne_bin(velo_path)
    objects = read_label_file(label_path)
    calib = read_calib_file(calib_path)

    points_cam = transform_velodyne_to_rect_camera(points_velo, calib)

    bev, roi_mask = make_bev_occupancy(points_velo)

    total_object_points = 0
    boxes_drawn = 0

    print()
    print("velodyne shape:", points_velo.shape)
    print("objects:", len(objects))
    print("ROI:")
    print("x:", X_MIN, X_MAX)
    print("y:", Y_MIN, Y_MAX)
    print("z:", Z_MIN, Z_MAX)
    print("resolution:", RESOLUTION)

    for idx, obj in enumerate(objects):
        cls = obj["type"]

        if cls == "DontCare":
            continue

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        corners_cam = compute_3d_box_corners_camera(obj)
        corners_velo = transform_rect_camera_to_velodyne(corners_cam, calib)

        ok = draw_bev_box(bev, corners_velo, color, thickness=2)

        if ok:
            boxes_drawn += 1

        inside = points_inside_3d_box_camera(points_cam, obj)
        object_points_velo = points_velo[inside, :3]

        total_object_points += len(object_points_velo)

        draw_bev_points(bev, object_points_velo, (0, 0, 255), radius=2)

        center_velo = transform_rect_camera_to_velodyne(
            obj["location_xyz_camera"].reshape(1, 3),
            calib,
        )[0]

        print()
        print("object", idx)
        print("type:", cls)
        print("camera location:", obj["location_xyz_camera"])
        print("velodyne center approx:", center_velo)
        print("dimensions h,w,l:", obj["dimensions_hwl"])
        print("rotation_y:", obj["rotation_y"])
        print("box drawn in BEV:", ok)
        print("LiDAR points inside 3D box:", len(object_points_velo))

    bev = add_title(bev, f"BEV GT 3D Boxes | Frame {FRAME_ID}")
    bev = add_footer(
        bev,
        f"gray=LiDAR ROI | colored boxes=KITTI GT | red=points inside 3D boxes | boxes={boxes_drawn} pts={total_object_points}",
    )

    out_path = OUTPUT_DIR / f"bev_gt_3d_boxes_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), bev)

    print()
    print("boxes drawn:", boxes_drawn)
    print("total LiDAR points inside boxes:", total_object_points)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
