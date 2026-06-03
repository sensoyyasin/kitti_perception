from pathlib import Path
import numpy as np
import cv2


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"
FRAME_ID = "000613"

OUTPUT_DIR = Path("../outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def transform_velodyne_to_rect_camera(points_velo, calib):
    """
    KITTI object transform:

    p_rect_cam = R0_rect @ Tr_velo_to_cam @ p_velo
    """
    R0_rect = calib["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_rect_4x4 = make_4x4_from_3x3(R0_rect)
    Tr_velo_to_cam_4x4 = make_4x4_from_3x4(Tr_velo_to_cam)

    xyz = points_velo[:, :3].astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.hstack([xyz, ones])

    points_cam_rect_h = (R0_rect_4x4 @ Tr_velo_to_cam_4x4 @ xyz_h.T).T

    return points_cam_rect_h[:, :3]


def project_camera_points_to_image(points_cam, P2, image_width, image_height):
    n = points_cam.shape[0]
    points_h = np.hstack([points_cam, np.ones((n, 1), dtype=np.float64)])

    pixels_h = (P2 @ points_h.T).T

    depth = pixels_h[:, 2]
    valid_depth = depth > 0.1

    u = pixels_h[:, 0] / pixels_h[:, 2]
    v = pixels_h[:, 1] / pixels_h[:, 2]

    inside = (
        valid_depth &
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )

    return u[inside].astype(np.int32), v[inside].astype(np.int32), depth[inside], inside


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
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    corners_cam = R @ corners_obj
    corners_cam[0, :] += x
    corners_cam[1, :] += y
    corners_cam[2, :] += z

    return corners_cam.T


def points_inside_3d_box_camera(points_cam, obj):
    """
    Test whether camera-frame points are inside KITTI 3D box.

    KITTI object box convention:
      x extent: [-l/2, l/2]
      y extent: [-h, 0]
      z extent: [-w/2, w/2]

    location is at bottom center of the object.
    rotation_y rotates around camera Y axis.
    """
    h, w, l = obj["dimensions_hwl"]
    center = obj["location_xyz_camera"].astype(np.float64)
    ry = obj["rotation_y"]

    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    # Move points into object local coordinate frame.
    points_local = (R.T @ (points_cam - center).T).T

    x = points_local[:, 0]
    y = points_local[:, 1]
    z = points_local[:, 2]

    inside = (
        (x >= -l / 2) & (x <= l / 2) &
        (y >= -h) & (y <= 0) &
        (z >= -w / 2) & (z <= w / 2)
    )

    return inside, points_local


def draw_3d_box(image, corners_2d, color, thickness=2):
    corners_2d = corners_2d.astype(int)

    bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]

    for i, j in bottom_edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)

    for i, j in top_edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)

    for i, j in vertical_edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)


def draw_2d_box(image, obj, color):
    x1, y1, x2, y2 = obj["bbox"].astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)


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

    image_path = root / "image_2" / f"{FRAME_ID}.png"
    velo_path = root / "velodyne" / f"{FRAME_ID}.bin"
    label_path = root / "label_2" / f"{FRAME_ID}.txt"
    calib_path = root / "calib" / f"{FRAME_ID}.txt"

    print("=== 14_6 LiDAR Points Inside 3D Boxes ===")
    print("frame:", FRAME_ID)
    print("image:", image_path)
    print("velodyne:", velo_path)
    print("label:", label_path)
    print("calib:", calib_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    points_velo = read_velodyne_bin(velo_path)
    objects = read_label_file(label_path)
    calib = read_calib_file(calib_path)

    P2 = calib["P2"].reshape(3, 4)

    points_cam = transform_velodyne_to_rect_camera(points_velo, calib)

    projected_all = image.copy()
    projected_inside_boxes = image.copy()

    all_object_point_count = 0

    print()
    print("image shape:", image.shape)
    print("velodyne shape:", points_velo.shape)
    print("objects:", len(objects))

    for obj_idx, obj in enumerate(objects):
        cls = obj["type"]

        if cls == "DontCare":
            continue

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        corners_cam = compute_3d_box_corners_camera(obj)
        corners_u, corners_v, _, valid_corners = project_camera_points_to_image(
            corners_cam,
            P2,
            width,
            height,
        )

        if len(corners_u) == 8:
            corners_2d = np.column_stack([corners_u, corners_v])
            draw_3d_box(projected_all, corners_2d, color, 2)
            draw_3d_box(projected_inside_boxes, corners_2d, color, 2)

        draw_2d_box(projected_all, obj, color)
        draw_2d_box(projected_inside_boxes, obj, color)

        inside_box, points_local = points_inside_3d_box_camera(points_cam, obj)
        object_points_cam = points_cam[inside_box]

        all_object_point_count += len(object_points_cam)

        u, v, depth, inside_image = project_camera_points_to_image(
            object_points_cam,
            P2,
            width,
            height,
        )

        for px, py in zip(u, v):
            cv2.circle(projected_inside_boxes, (int(px), int(py)), 4, (0, 0, 255), -1)

        x1, y1, x2, y2 = obj["bbox"].astype(int)
        label = f"{cls} 3Dpts={len(object_points_cam)} z={obj['location_xyz_camera'][2]:.1f}m"

        cv2.putText(
            projected_all,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            projected_inside_boxes,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        print()
        print("object", obj_idx)
        print("type:", cls)
        print("2D bbox:", obj["bbox"])
        print("3D location camera:", obj["location_xyz_camera"])
        print("dimensions h,w,l:", obj["dimensions_hwl"])
        print("rotation_y:", obj["rotation_y"])
        print("LiDAR points inside 3D box:", len(object_points_cam))

        if len(object_points_cam) > 0:
            print("inside-box camera depth min:", float(object_points_cam[:, 2].min()))
            print("inside-box camera depth mean:", float(object_points_cam[:, 2].mean()))
            print("inside-box camera depth max:", float(object_points_cam[:, 2].max()))

    raw = add_title(image, f"Raw image_2/{FRAME_ID}.png")
    raw = add_footer(raw, "original image")

    projected_all = add_title(projected_all, "GT 2D + projected GT 3D boxes")
    projected_all = add_footer(projected_all, "3D boxes come from KITTI label_2")

    projected_inside_boxes = add_title(projected_inside_boxes, "LiDAR points inside GT 3D boxes")
    projected_inside_boxes = add_footer(
        projected_inside_boxes,
        f"red points are Velodyne returns inside 3D boxes | total object pts={all_object_point_count}",
    )

    combined = np.vstack([
        np.hstack([raw, projected_all]),
        np.hstack([image, projected_inside_boxes]),
    ])

    out_path = OUTPUT_DIR / f"lidar_inside_3d_boxes_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), combined)

    print()
    print("total LiDAR points inside all 3D boxes:", all_object_point_count)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
