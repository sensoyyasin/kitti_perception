import os
import numpy as np
import cv2
import pykitti
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


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

    return velo[mask]


def metric_to_bev_pixels(points, x_min, x_max, y_min, y_max, resolution):
    x = points[:, 0]
    y = points[:, 1]

    rows = ((x_max - x) / resolution).astype(np.int32)

    # KITTI: y positive = vehicle-left.
    # Map vehicle-left to BEV-left.
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

    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    return np.array([a, b, c, d], dtype=np.float64)


def signed_distance_to_plane(points, plane):
    a, b, c, d = plane
    denom = np.sqrt(a * a + b * b + c * c)

    return (
        a * points[:, 0]
        + b * points[:, 1]
        + c * points[:, 2]
        + d
    ) / denom


def ransac_plane(points, num_iterations=350, distance_threshold=0.08, seed=42):
    rng = np.random.default_rng(seed)
    n = len(points)

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
            best_plane = plane
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count

    return best_plane, best_inlier_mask


def refine_plane_least_squares(points):
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


def classify_ground_global_ransac(velo, cfg, frame_idx):
    x_min, x_max, y_min, y_max, z_min, z_max, resolution = cfg

    velo_roi = filter_roi(
        velo,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )

    points = velo_roi[:, :3]
    class_ids = np.zeros(len(points), dtype=np.int32)
    residuals = np.full(len(points), np.nan, dtype=np.float32)

    if len(points) < 100:
        return points, class_ids, residuals, {"status": "not enough points"}

    z = points[:, 2]

    # Stable KITTI road candidate height range.
    candidate_mask = (z > -2.2) & (z < -1.0)
    fit_points = points[candidate_mask]

    if len(fit_points) < 500:
        return points, class_ids, residuals, {"status": "not enough fit points"}

    plane, inlier_mask = ransac_plane(
        fit_points,
        num_iterations=350,
        distance_threshold=0.08,
        seed=frame_idx + 42,
    )

    if plane is None or inlier_mask is None or np.sum(inlier_mask) < 100:
        return points, class_ids, residuals, {"status": "RANSAC failed"}

    refined_plane = refine_plane_least_squares(fit_points[inlier_mask])
    residuals = signed_distance_to_plane(points, refined_plane).astype(np.float32)

    ground_threshold = 0.10
    above_threshold = 0.20
    below_threshold = -0.16

    ground = np.abs(residuals) <= ground_threshold
    above = residuals > above_threshold
    below = residuals < below_threshold

    class_ids[ground] = 1
    class_ids[above] = 2
    class_ids[below] = 3

    stats = {
        "status": "ok",
        "roi_points": int(len(points)),
        "fit_points": int(len(fit_points)),
        "inliers": int(np.sum(inlier_mask)),
        "ground": int(np.sum(ground)),
        "above": int(np.sum(above)),
        "below": int(np.sum(below)),
        "plane": refined_plane,
    }

    return points, class_ids, residuals, stats


def knn_filter_points(points_xy, k=8, max_mean_dist=0.65):
    """
    Remove isolated below points using mean kNN distance.
    """
    if len(points_xy) < k + 1:
        return np.zeros(len(points_xy), dtype=bool)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(points_xy)

    distances, _ = nn.kneighbors(points_xy)

    # Ignore self-distance at column 0.
    mean_dist = distances[:, 1:].mean(axis=1)

    keep = mean_dist <= max_mean_dist
    return keep


def cluster_below_points(points, residuals):
    """
    Cluster below-surface points.

    Returns:
        cluster_labels_full:
            length N array, -1 for non-cluster or non-below.
        clusters:
            list of cluster metric dicts.
    """
    cluster_labels_full = np.full(len(points), -1, dtype=np.int32)

    # Below-surface candidate points.
    below_threshold = -0.10

    x = points[:, 0]
    y = points[:, 1]

    corridor = (
        (y >= -3.0) & (y <= 3.0) &
        (x >= 5.0) & (x <= 30.0)
    )

    below_mask = (residuals < below_threshold) & corridor
    below_idx = np.where(below_mask)[0]

    if len(below_idx) < 10:
        return cluster_labels_full, []

    below_points = points[below_idx]
    below_xy = below_points[:, :2]

    # KNN filter
    keep_knn = knn_filter_points(
        below_xy,
        k=8,
        max_mean_dist=0.65,
    )

    filtered_idx = below_idx[keep_knn]

    if len(filtered_idx) < 10:
        return cluster_labels_full, []

    filtered_points = points[filtered_idx]
    filtered_xy = filtered_points[:, :2]

    # DBSCAN clustering in x-y ground plane.
    db = DBSCAN(
        eps=0.45,
        min_samples=10,
    )

    labels = db.fit_predict(filtered_xy)

    clusters = []
    next_cluster_id = 0

    for label in sorted(set(labels)):
        if label == -1:
            continue

        local = labels == label
        cluster_idx = filtered_idx[local]
        cluster_points = points[cluster_idx]
        cluster_res = residuals[cluster_idx]

        point_count = len(cluster_idx)

        if point_count < 20:
            continue

        xs = cluster_points[:, 0]
        ys = cluster_points[:, 1]

        x_extent = float(xs.max() - xs.min())
        y_extent = float(ys.max() - ys.min())

        # Reject huge road-edge artifacts.
        if x_extent > 8.0 or y_extent > 5.0:
            continue

        max_depth = float(-np.min(cluster_res))
        mean_depth = float(-np.mean(cluster_res))

        if max_depth < 0.10:
            continue

        cluster_labels_full[cluster_idx] = next_cluster_id

        clusters.append({
            "cluster_id": int(next_cluster_id),
            "point_count": int(point_count),
            "center_x": float(xs.mean()),
            "center_y": float(ys.mean()),
            "x_min": float(xs.min()),
            "x_max": float(xs.max()),
            "y_min": float(ys.min()),
            "y_max": float(ys.max()),
            "x_extent": x_extent,
            "y_extent": y_extent,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
        })

        next_cluster_id += 1

    return cluster_labels_full, clusters


def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.hstack([points_xyz, ones])


def project_points_to_camera(points_velo, K, T_cam2_velo, width, height):
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

    valid_z = Z > 0

    u = np.zeros_like(Z)
    v = np.zeros_like(Z)

    u[valid_z] = fx * X[valid_z] / Z[valid_z] + cx
    v[valid_z] = fy * Y[valid_z] / Z[valid_z] + cy

    valid = (
        valid_z &
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    return u[valid].astype(np.int32), v[valid].astype(np.int32), valid


def draw_bev(points, class_ids, cluster_labels, clusters, cfg, frame_idx, stats):
    x_min, x_max, y_min, y_max, z_min, z_max, resolution = cfg

    rows, cols, valid, h, w = metric_to_bev_pixels(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution,
    )

    class_valid = class_ids[valid]
    cluster_valid = cluster_labels[valid]

    bev = np.zeros((h, w, 3), dtype=np.uint8)

    colors = {
        0: (90, 90, 90),      # unknown
        1: (60, 220, 60),     # ground
        2: (40, 120, 255),    # above
        3: (255, 80, 40),     # raw below blue
    }

    for cid, color in colors.items():
        m = class_valid == cid
        bev[rows[m], cols[m]] = color

    # Clustered below points = red/yellow.
    clustered = cluster_valid >= 0
    bev[rows[clustered], cols[clustered]] = (0, 0, 255)

    # Draw cluster bounding boxes in BEV.
    for c in clusters:
        box_points = np.array([
            [c["x_min"], c["y_min"], 0.0],
            [c["x_min"], c["y_max"], 0.0],
            [c["x_max"], c["y_max"], 0.0],
            [c["x_max"], c["y_min"], 0.0],
        ], dtype=np.float32)

        br, bc, bv, _, _ = metric_to_bev_pixels(
            box_points,
            x_min,
            x_max,
            y_min,
            y_max,
            resolution,
        )

        if len(br) == 4:
            pts = np.column_stack([bc, br]).astype(np.int32)
            cv2.polylines(bev, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            label_pos = tuple(pts[0])
            cv2.putText(
                bev,
                f"C{c['cluster_id']}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Ego corridor rectangle.
    corridor_corners = np.array([
        [3.0, -5.0, 0.0],
        [3.0, 5.0, 0.0],
        [40.0, 5.0, 0.0],
        [40.0, -5.0, 0.0],
    ], dtype=np.float32)

    rr, cc, vv, _, _ = metric_to_bev_pixels(
        corridor_corners,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution,
    )

    if len(rr) == 4:
        pts = np.column_stack([cc, rr]).astype(np.int32)
        cv2.polylines(bev, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.putText(
        bev,
        f"Below-Surface Clusters | Frame {frame_idx}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    line = (
        f"ground={stats.get('ground', 0)} "
        f"above={stats.get('above', 0)} "
        f"below={stats.get('below', 0)} "
        f"clusters={len(clusters)}"
    )

    cv2.putText(
        bev,
        line,
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return bev


def draw_camera(img_cam2, points, class_ids, cluster_labels, K, T_cam2_velo, stats, clusters):
    img_rgb = np.array(img_cam2)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    height, width = img_bgr.shape[:2]

    u, v, valid = project_points_to_camera(
        points,
        K,
        T_cam2_velo,
        width,
        height,
    )

    class_valid = class_ids[valid]
    cluster_valid = cluster_labels[valid]

    colors = {
        1: (60, 220, 60),     # ground
        2: (40, 120, 255),    # above
        3: (255, 80, 40),     # raw below
    }

    # Draw ground and above lightly.
    for cid in [1, 2, 3]:
        mask = class_valid == cid
        uu = u[mask]
        vv = v[mask]

        radius = 1
        if cid == 3:
            radius = 2

        for px, py in zip(uu, vv):
            cv2.circle(img_bgr, (px, py), radius, colors[cid], -1)

    # Draw clustered below points on top in red.
    clustered = cluster_valid >= 0
    for px, py in zip(u[clustered], v[clustered]):
        cv2.circle(img_bgr, (px, py), 3, (0, 0, 255), -1)

    cv2.putText(
        img_bgr,
        "Camera: Below-Surface Residual Clusters",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img_bgr,
        "blue=raw below | red=DBSCAN clusters",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    line = f"clusters={len(clusters)} below={stats.get('below', 0)}"

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
        0.0,
        50.0,
        -25.0,
        25.0,
        -3.0,
        2.0,
        0.10,
    )

    output_path = "../outputs/videos/below_surface_clusters_video.mp4"

    target_height = 500
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    for i in range(NUM_FRAMES):
        img_cam2, _ = data.get_rgb(i)
        velo = data.get_velo(i)

        points, class_ids, residuals, stats = classify_ground_global_ransac(
            velo,
            cfg,
            i,
        )

        cluster_labels, clusters = cluster_below_points(points, residuals)

        camera_view = draw_camera(
            img_cam2,
            points,
            class_ids,
            cluster_labels,
            K,
            T_cam2_velo,
            stats,
            clusters,
        )

        bev_view = draw_bev(
            points,
            class_ids,
            cluster_labels,
            clusters,
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
            print(
                f"processed frame {i}/{NUM_FRAMES}, "
                f"below={stats.get('below', 0)}, "
                f"clusters={len(clusters)}"
            )

    writer.release()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
