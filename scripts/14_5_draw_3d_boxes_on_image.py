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


def compute_3d_box_corners_camera(obj):
    """
    KITTI object label convention:
    dimensions are h, w, l
    location is object center in camera coordinates, but approximately at bottom center.
    rotation_y is yaw around camera Y axis.

    Camera coordinates:
      x = right
      y = down
      z = forward
    """
    h, w, l = obj["dimensions_hwl"]
    x, y, z = obj["location_xyz_camera"]
    ry = obj["rotation_y"]

    # 3D bounding box corners in object coordinate frame.
    # KITTI convention: y=0 is bottom, y=-h is top.
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

    # Rotation around camera Y axis.
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


def project_camera_points_to_image(points_cam, P2):
    """
    points_cam: Nx3 camera coordinates
    P2: 3x4 projection matrix
    """
    n = points_cam.shape[0]
    points_h = np.hstack([points_cam, np.ones((n, 1))])

    pixels_h = (P2 @ points_h.T).T

    u = pixels_h[:, 0] / pixels_h[:, 2]
    v = pixels_h[:, 1] / pixels_h[:, 2]
    z = pixels_h[:, 2]

    pixels = np.column_stack([u, v])

    return pixels, z


def draw_2d_box(image, obj, color):
    x1, y1, x2, y2 = obj["bbox"].astype(int)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        1,
    )


def draw_3d_box(image, corners_2d, color, thickness=2):
    """
    Corner order:
      0-3 bottom rectangle
      4-7 top rectangle
    """
    corners_2d = corners_2d.astype(int)

    # bottom face
    bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # top face
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]

    # vertical edges
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]

    for i, j in bottom_edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)

    for i, j in top_edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)

    for i, j in vertical_edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)

    # Draw front face indicator.
    # In this corner convention, the front face is often the first two bottom/top points,
    # but for debugging, draw a diagonal on one face.
    cv2.line(image, tuple(corners_2d[0]), tuple(corners_2d[5]), color, 1)


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
    label_path = root / "label_2" / f"{FRAME_ID}.txt"
    calib_path = root / "calib" / f"{FRAME_ID}.txt"

    print("=== 14_5 Draw 3D Boxes on Image ===")
    print("frame:", FRAME_ID)
    print("image:", image_path)
    print("label:", label_path)
    print("calib:", calib_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    objects = read_label_file(label_path)
    calib = read_calib_file(calib_path)

    P2 = calib["P2"].reshape(3, 4)

    raw = image.copy()
    overlay = image.copy()

    print()
    print("objects:", len(objects))

    for i, obj in enumerate(objects):
        cls = obj["type"]

        if cls == "DontCare":
            continue

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        corners_cam = compute_3d_box_corners_camera(obj)

        # Skip boxes behind camera.
        if np.any(corners_cam[:, 2] <= 0.1):
            print(i, cls, "skipped, box partly behind camera")
            continue

        corners_2d, depths = project_camera_points_to_image(corners_cam, P2)

        draw_2d_box(overlay, obj, color)
        draw_3d_box(overlay, corners_2d, color, thickness=2)

        x1, y1, x2, y2 = obj["bbox"].astype(int)

        label = f"{cls} z={obj['location_xyz_camera'][2]:.1f}m"

        cv2.putText(
            overlay,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        print()
        print("object", i)
        print("type:", cls)
        print("2D bbox:", obj["bbox"])
        print("3D dimensions h,w,l:", obj["dimensions_hwl"])
        print("3D location camera:", obj["location_xyz_camera"])
        print("rotation_y:", obj["rotation_y"])
        print("3D corners camera:")
        print(corners_cam)
        print("projected corners 2D:")
        print(corners_2d)

    raw = add_title(raw, f"Raw image_2/{FRAME_ID}.png")
    overlay = add_title(overlay, "KITTI 2D boxes + projected 3D boxes")

    combined = np.hstack([raw, overlay])

    out_path = OUTPUT_DIR / f"object_3d_boxes_image_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), combined)

    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
