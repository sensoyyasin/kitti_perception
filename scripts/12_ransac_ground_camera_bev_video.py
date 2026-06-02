import os
import numpy as np
import cv2
import pykitti

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"
NUM_FRAMES = 481

def filter_roi(velo, x_min, x_max, y_min, y_max, z_min, z_max):
    x = velo[:, 0]
    y = velo[:, 1]
    z = velo[:, 2]

    mask = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z < z_max)
    )

    return velo[mask], mask


def metric_to_bev_pixels(points, x_min, x_max, y_min, y_max, resolution):
    x = points[:, 0]
    y = points[:, 1]

    rows = ((x_max - x) / resolution).astype(np.int32)

    # KITTI y positive = vehicle-left.
    # Map vehicle-left to image-left.
    cols = ((y_max - y) / resolution).astype(np.int32)

    bev_height = int((x_max - x_min) / resolution)
    bev_width = int((y_max - y_min) / resolution)

    valid = (
        (rows >= 0) & (rows < bev_height) &
        (cols >= 0) & (cols < bev_width)
    )

    return rows[valid], cols[valid], valid, bev_height, bev_width


def plane_from_3_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return None

    normal = normal / norm
    a, b, c = normal
    d = -np.dot(normal, p1)

    # KITTI z is up. Keep normal pointing upward.
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    return np.array([a, b, c, d], dtype=np.float64)


def signed_distance_to_plane(points, plane):
    a, b, c, d = plane
    denom = np.sqrt(a * a + b * b + c * c)
    return (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / denom


def ransac_plane(points, num_iterations=350, distance_threshold=0.08, seed=42):
    rng = np.random.default_rng(seed)

    n = points.shape[0]

    if n < 3:
        return None, None

    best_plane = None
    best_inlier_mask = None
    best_inlier_count = 0

    for _ in range(num_iterations):
        ids = rng.choice(n, size=3, replace=False)

        plane = plane_from_3_points(
            points[ids[0]],
            points[ids[1]],
            points[ids[2]],
        )

        if plane is None:
            continue

        dist = signed_distance_to_plane(points, plane)
        inlier_mask = np.abs(dist) < distance_threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inlier_mask = inlier_mask

    return best_plane, best_inlier_mask


def refine_plane_least_squares(points):
    """
    Fit z = alpha*x + beta*y + gamma using least squares.
    Convert to plane: alpha*x + beta*y - z + gamma = 0.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    alpha, beta, gamma = coeffs

    plane = np.array([alpha, beta, -1.0, gamma], dtype=np.float64)
    plane = plane / np.linalg.norm(plane[:3])

    if plane[2] < 0:
        plane = -plane

    return plane


def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.hstack([points_xyz, ones])


def project_points_to_camera(points_velo, K, T_cam2_velo, width, height):
    """
    Project arbitrary Velodyne points to cam2 image.
    Returns projected u,v and valid mask relative to input points.
    """
    points_h = to_homogeneous(points_velo)
    points_cam_h = (T_cam2_velo @ points_h.T).T
    points_cam = points_cam_h[:, :3]

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    valid = (
        (Z > 0) &
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    return u[valid].astype(np.int32), v[valid].astype(np.int32), valid

def classify_ground_with_ransac(velo, cfg, frame_idx):
    """
    Returns:
        points_roi: Nx3
        class_ids: N int
            0 = unclassified
            1 = ground
            2 = above
            3 = below
        residuals: N float
        stats: dict
    """
    x_min, x_max, y_min, y_max, z_min, z_max, resolution = cfg

    velo_roi, _ = filter_roi(
        velo,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )

    points_roi = velo_roi[:, :3]

    class_ids = np.zeros(len(points_roi), dtype=np.int32)
    residuals = np.full(len(points_roi), np.nan, dtype=np.float32)

    if len(points_roi) < 100:
        stats = {"status": "not enough ROI points"}
        return points_roi, class_ids, residuals, stats

    z = points_roi[:, 2]

    # Simple initial road candidate selection.
    candidate_mask = (z > -2.2) & (z < -1.0)
    fit_points = points_roi[candidate_mask]

    if len(fit_points) < 500:
        stats = {"status": "not enough fit points"}
        return points_roi, class_ids, residuals, stats

    plane, inlier_mask = ransac_plane(
        fit_points,
        num_iterations=350,
        distance_threshold=0.08,
        seed=frame_idx + 42,
    )

    if plane is None or inlier_mask is None or np.sum(inlier_mask) < 100:
        stats = {"status": "RANSAC failed"}
        return points_roi, class_ids, residuals, stats

    refined_plane = refine_plane_least_squares(fit_points[inlier_mask])

    residuals = signed_distance_to_plane(points_roi, refined_plane).astype(np.float32)

    ground_threshold = 0.10
    above_threshold = 0.20
    below_threshold = -0.10

    ground_mask = np.abs(residuals) <= ground_threshold
    above_mask = residuals > above_threshold
    below_mask = residuals < below_threshold

    class_ids[ground_mask] = 1
    class_ids[above_mask] = 2
    class_ids[below_mask] = 3

    stats = {
        "status": "ok",
        "roi_points": int(len(points_roi)),
        "fit_candidates": int(len(fit_points)),
        "inliers": int(np.sum(inlier_mask)),
        "ground": int(np.sum(ground_mask)),
        "above": int(np.sum(above_mask)),
        "below": int(np.sum(below_mask)),
        "plane": refined_plane,
        "residual_min": float(np.nanmin(residuals)),
        "residual_mean": float(np.nanmean(residuals)),
        "residual_max": float(np.nanmax(residuals)),
    }

    return points_roi, class_ids, residuals, stats


# -----------------------------
# Drawing utilities
# -----------------------------

def draw_bev_classes(points_roi, class_ids, cfg, frame_idx, stats):
    x_min, x_max, y_min, y_max, z_min, z_max, resolution = cfg

    rows, cols, valid, h, w = metric_to_bev_pixels(
        points_roi,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution,
    )

    class_valid = class_ids[valid]

    bev = np.zeros((h, w, 3), dtype=np.uint8)

    # BGR colors
    color_ground = (60, 220, 60)
    color_above = (40, 120, 255)
    color_below = (255, 80, 40)
    color_unknown = (100, 100, 100)

    unknown = class_valid == 0
    ground = class_valid == 1
    above = class_valid == 2
    below = class_valid == 3

    bev[rows[unknown], cols[unknown]] = color_unknown
    bev[rows[ground], cols[ground]] = color_ground
    bev[rows[above], cols[above]] = color_above
    bev[rows[below], cols[below]] = color_below

    cv2.putText(
        bev,
        f"BEV RANSAC Ground | Frame {frame_idx}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        bev,
        "green=ground | orange=above | blue=below",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if stats.get("status") == "ok":
        line = (
            f"ground={stats['ground']} "
            f"above={stats['above']} "
            f"below={stats['below']}"
        )
    else:
        line = stats.get("status", "unknown")

    cv2.putText(
        bev,
        line,
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return bev


def draw_camera_classes(img_cam2, points_roi, class_ids, K, T_cam2_velo, stats):
    img_rgb = np.array(img_cam2)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    height, width = img_bgr.shape[:2]

    u, v, valid = project_points_to_camera(
        points_roi,
        K,
        T_cam2_velo,
        width,
        height,
    )

    class_valid = class_ids[valid]

    # Draw order: ground first, above next, below last.
    # This helps blue points stay visible.
    draw_order_classes = [1, 2, 3, 0]

    colors = {
        0: (160, 160, 160),  # unknown
        1: (60, 220, 60),    # ground green
        2: (40, 120, 255),   # above orange/red
        3: (255, 80, 40),    # below blue
    }

    for cid in draw_order_classes:
        mask = class_valid == cid
        uu = u[mask]
        vv = v[mask]
        color = colors[cid]

        radius = 1
        if cid == 3:
            radius = 2

        for px, py in zip(uu, vv):
            cv2.circle(img_bgr, (px, py), radius, color, -1)

    cv2.putText(
        img_bgr,
        "Camera projection of RANSAC classes",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img_bgr,
        "green=ground | orange=above | blue=below",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if stats.get("status") == "ok":
        line = (
            f"ground={stats['ground']} "
            f"above={stats['above']} "
            f"below={stats['below']}"
        )
    else:
        line = stats.get("status", "unknown")

    cv2.putText(
        img_bgr,
        line,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return img_bgr


def resize_to_height(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs("../outputs/videos", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(NUM_FRAMES))

    duration = (data.timestamps[-1] - data.timestamps[0]).total_seconds()
    fps = (len(data.timestamps) - 1) / duration

    print("Using FPS:", fps)

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    cfg = (
        0.0,     # x_min
        50.0,    # x_max
        -25.0,   # y_min
        25.0,    # y_max
        -3.0,    # z_min
        2.0,     # z_max
        0.10,    # resolution
    )

    output_path = "../outputs/videos/ransac_ground_camera_bev_video.mp4"

    target_height = 500

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for i in range(NUM_FRAMES):
        img_cam2, _ = data.get_rgb(i)
        velo = data.get_velo(i)

        points_roi, class_ids, residuals, stats = classify_ground_with_ransac(
            velo,
            cfg,
            i,
        )

        camera_view = draw_camera_classes(
            img_cam2,
            points_roi,
            class_ids,
            K,
            T_cam2_velo,
            stats,
        )

        bev_view = draw_bev_classes(
            points_roi,
            class_ids,
            cfg,
            i,
            stats,
        )

        camera_resized = resize_to_height(camera_view, target_height)
        bev_resized = resize_to_height(bev_view, target_height)

        combined = np.hstack([camera_resized, bev_resized])

        if writer is None:
            h, w = combined.shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            if not writer.isOpened():
                raise RuntimeError("Could not open VideoWriter.")

        writer.write(combined)

        if i % 25 == 0:
            if stats.get("status") == "ok":
                print(
                    f"processed frame {i}/{NUM_FRAMES}, "
                    f"ground={stats['ground']}, "
                    f"above={stats['above']}, "
                    f"below={stats['below']}"
                )
            else:
                print(f"processed frame {i}/{NUM_FRAMES}, {stats.get('status')}")

    writer.release()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
