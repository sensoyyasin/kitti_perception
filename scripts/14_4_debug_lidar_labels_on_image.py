from pathlib import Path
import numpy as np
import cv2


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"
FRAME_ID = "000000"

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
    points = points.reshape(-1, 4)
    return points


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


def depth_to_bgr(depth, max_depth=80.0):
    d = np.clip(depth, 0.0, max_depth)
    ratio = d / max_depth

    red = int(255 * (1.0 - ratio))
    green = int(255 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(255 * ratio)

    return (blue, green, red)


def project_velodyne_to_image(points_velo, calib, image_width, image_height):
    """
    KITTI object projection:

    pixel_h = P2 @ R0_rect @ Tr_velo_to_cam @ point_velo_h

    Returns projected pixel coordinates, depth, reflectance, and original indices.
    """
    P2 = calib["P2"].reshape(3, 4)
    R0_rect = calib["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_rect_4x4 = make_4x4_from_3x3(R0_rect)
    Tr_velo_to_cam_4x4 = make_4x4_from_3x4(Tr_velo_to_cam)

    xyz = points_velo[:, :3].astype(np.float64)
    reflectance = points_velo[:, 3]

    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.hstack([xyz, ones])

    points_cam_rect_h = (R0_rect_4x4 @ Tr_velo_to_cam_4x4 @ xyz_h.T).T

    depth = points_cam_rect_h[:, 2]

    in_front = depth > 0.1
    original_indices_front = np.where(in_front)[0]

    points_cam_rect_h_front = points_cam_rect_h[in_front]
    depth_front = depth[in_front]
    reflectance_front = reflectance[in_front]

    pixels_h = (P2 @ points_cam_rect_h_front.T).T

    u = pixels_h[:, 0] / pixels_h[:, 2]
    v = pixels_h[:, 1] / pixels_h[:, 2]

    inside = (
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )

    return {
        "u": u[inside].astype(np.int32),
        "v": v[inside].astype(np.int32),
        "depth": depth_front[inside],
        "reflectance": reflectance_front[inside],
        "original_indices": original_indices_front[inside],
        "num_total": len(points_velo),
        "num_in_front": int(np.sum(in_front)),
        "num_inside": int(np.sum(inside)),
    }


def draw_lidar_points(image_bgr, proj, radius=1):
    out = image_bgr.copy()

    u = proj["u"]
    v = proj["v"]
    depth = proj["depth"]

    order = np.argsort(depth)[::-1]

    for idx in order:
        px = int(u[idx])
        py = int(v[idx])
        d = float(depth[idx])

        color = depth_to_bgr(d)
        cv2.circle(out, (px, py), radius, color, -1)

    return out


def compute_bbox_lidar_stats(objects, proj):
    u = proj["u"]
    v = proj["v"]
    depth = proj["depth"]
    reflectance = proj["reflectance"]

    stats = []

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]

        inside = (
            (u >= x1) & (u <= x2) &
            (v >= y1) & (v <= y2)
        )

        count = int(np.sum(inside))

        if count > 0:
            d = depth[inside]
            r = reflectance[inside]

            obj_stats = {
                "count": count,
                "depth_min": float(np.min(d)),
                "depth_mean": float(np.mean(d)),
                "depth_max": float(np.max(d)),
                "refl_mean": float(np.mean(r)),
            }
        else:
            obj_stats = {
                "count": 0,
                "depth_min": None,
                "depth_mean": None,
                "depth_max": None,
                "refl_mean": None,
            }

        stats.append(obj_stats)

    return stats


def draw_boxes_and_stats(image_bgr, objects, bbox_stats):
    out = image_bgr.copy()

    for obj, st in zip(objects, bbox_stats):
        cls = obj["type"]
        x1, y1, x2, y2 = obj["bbox"].astype(int)

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        thickness = 2
        if cls == "DontCare":
            thickness = 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if st["count"] > 0:
            label = f"{cls} | pts={st['count']} | z={st['depth_mean']:.1f}m"
        else:
            label = f"{cls} | pts=0"

        cv2.putText(
            out,
            label,
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_only_bbox_lidar_points(image_bgr, objects, proj):
    """
    Draw only LiDAR points that fall inside any 2D label bbox.
    Useful to see object-associated LiDAR returns.
    """
    out = image_bgr.copy()

    u = proj["u"]
    v = proj["v"]
    depth = proj["depth"]

    any_inside = np.zeros(len(u), dtype=bool)

    for obj in objects:
        if obj["type"] == "DontCare":
            continue

        x1, y1, x2, y2 = obj["bbox"]

        inside = (
            (u >= x1) & (u <= x2) &
            (v >= y1) & (v <= y2)
        )

        any_inside |= inside

    selected_u = u[any_inside]
    selected_v = v[any_inside]
    selected_depth = depth[any_inside]

    order = np.argsort(selected_depth)[::-1]

    for idx in order:
        px = int(selected_u[idx])
        py = int(selected_v[idx])
        d = float(selected_depth[idx])

        color = depth_to_bgr(d)
        cv2.circle(out, (px, py), 3, color, -1)

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


def add_debug_footer(image, text):
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


def resize_to_width(image, target_width):
    h, w = image.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)

    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def main():
    root = BASEDIR / SPLIT

    image_path = root / "image_2" / f"{FRAME_ID}.png"
    velo_path = root / "velodyne" / f"{FRAME_ID}.bin"
    label_path = root / "label_2" / f"{FRAME_ID}.txt"
    calib_path = root / "calib" / f"{FRAME_ID}.txt"

    print("=== 14_4 Debug LiDAR + Labels on Image ===")
    print("frame:", FRAME_ID)
    print("image:", image_path)
    print("velodyne:", velo_path)
    print("label:", label_path)
    print("calib:", calib_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points = read_velodyne_bin(velo_path)
    objects = read_label_file(label_path)
    calib = read_calib_file(calib_path)

    height, width = image.shape[:2]

    proj = project_velodyne_to_image(points, calib, width, height)
    bbox_stats = compute_bbox_lidar_stats(objects, proj)

    print()
    print("image shape:", image.shape)
    print("velodyne shape:", points.shape)
    print("total velodyne points:", proj["num_total"])
    print("points in front of camera:", proj["num_in_front"])
    print("points projected inside image:", proj["num_inside"])

    if len(proj["depth"]) > 0:
        print("projected depth min:", float(np.min(proj["depth"])))
        print("projected depth mean:", float(np.mean(proj["depth"])))
        print("projected depth max:", float(np.max(proj["depth"])))

    print()
    print("objects and LiDAR points inside 2D bbox:")
    for i, (obj, st) in enumerate(zip(objects, bbox_stats)):
        print(
            i,
            obj["type"],
            "bbox:", obj["bbox"],
            "pts:", st["count"],
            "depth_mean:", st["depth_mean"],
            "depth_min:", st["depth_min"],
            "depth_max:", st["depth_max"],
        )

    raw = add_title(image, f"Raw image_2/{FRAME_ID}.png")
    raw = add_debug_footer(raw, f"objects={len(objects)}")

    label_overlay = draw_boxes_and_stats(image, objects, bbox_stats)
    label_overlay = add_title(label_overlay, "2D labels + bbox LiDAR stats")
    label_overlay = add_debug_footer(label_overlay, "bbox labels are ground truth from label_2")

    lidar_overlay = draw_lidar_points(image, proj, radius=1)
    lidar_overlay = draw_boxes_and_stats(lidar_overlay, objects, bbox_stats)
    lidar_overlay = add_title(lidar_overlay, "LiDAR projection + 2D labels")
    lidar_overlay = add_debug_footer(
        lidar_overlay,
        f"total={proj['num_total']} front={proj['num_in_front']} inside_img={proj['num_inside']}",
    )

    object_lidar = draw_only_bbox_lidar_points(image, objects, proj)
    object_lidar = draw_boxes_and_stats(object_lidar, objects, bbox_stats)
    object_lidar = add_title(object_lidar, "Only LiDAR points inside GT boxes")
    object_lidar = add_debug_footer(object_lidar, "debug view for object-associated LiDAR returns")

    # Make a 2x2 panel
    target_width = 620

    panels = [
        resize_to_width(raw, target_width),
        resize_to_width(label_overlay, target_width),
        resize_to_width(lidar_overlay, target_width),
        resize_to_width(object_lidar, target_width),
    ]

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    combined = np.vstack([top, bottom])

    out_path = OUTPUT_DIR / f"debug_lidar_labels_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), combined)

    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
