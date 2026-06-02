import os
import numpy as np
import cv2
import pykitti


BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"


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

    # KITTI y positive = vehicle left.
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
    """
    Return plane coefficients [a,b,c,d] for ax + by + cz + d = 0.
    """
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return None

    normal = normal / norm
    a, b, c = normal
    d = -np.dot(normal, p1)

    # Make normal point roughly upward.
    # KITTI z axis is up, so c should be positive.
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    return np.array([a, b, c, d], dtype=np.float64)


def signed_distance_to_plane(points, plane):
    a, b, c, d = plane
    denom = np.sqrt(a * a + b * b + c * c)
    return (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / denom


def ransac_plane(points, num_iterations=500, distance_threshold=0.08, seed=42):
    """
    Simple NumPy RANSAC plane fitting.
    """
    rng = np.random.default_rng(seed)

    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for plane fitting.")

    best_plane = None
    best_inlier_mask = None
    best_inlier_count = 0

    for _ in range(num_iterations):
        ids = rng.choice(n, size=3, replace=False)

        p1 = points[ids[0]]
        p2 = points[ids[1]]
        p3 = points[ids[2]]

        plane = plane_from_3_points(p1, p2, p3)

        if plane is None:
            continue

        dist = signed_distance_to_plane(points, plane)
        inlier_mask = np.abs(dist) < distance_threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inlier_mask = inlier_mask

    if best_plane is None:
        raise RuntimeError("RANSAC failed to find a valid plane.")

    return best_plane, best_inlier_mask


def refine_plane_least_squares(points):
    """
    Fit plane z = ax + by + c using least squares.
    Return plane [a,b,c,d] for ax + by + cz + d = 0.

    z = alpha*x + beta*y + gamma
    alpha*x + beta*y - z + gamma = 0
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    alpha, beta, gamma = coeffs

    # alpha*x + beta*y - z + gamma = 0
    plane = np.array([alpha, beta, -1.0, gamma], dtype=np.float64)

    # Normalize
    norm = np.linalg.norm(plane[:3])
    plane = plane / norm

    # Make normal z-positive
    if plane[2] < 0:
        plane = -plane

    return plane


def make_ransac_ground_bev(velo, cfg):
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

    points_roi = velo_roi[:, :3]

    # Candidate points for road fitting:
    # Use lower points only to avoid trees/cars dominating RANSAC.
    # KITTI road is usually around z = -1.7 to -1.3.
    z = points_roi[:, 2]
    candidate_mask = (z > -2.2) & (z < -1.0)
    fit_points = points_roi[candidate_mask]

    if len(fit_points) < 100:
        raise RuntimeError("Not enough candidate ground points for RANSAC.")

    plane, inlier_mask = ransac_plane(
        fit_points,
        num_iterations=700,
        distance_threshold=0.08,
        seed=42,
    )

    # Optional refinement on RANSAC inliers
    refined_plane = refine_plane_least_squares(fit_points[inlier_mask])

    residuals = signed_distance_to_plane(points_roi, refined_plane)

    # Classify
    ground_threshold = 0.10
    above_threshold = 0.20
    below_threshold = -0.10

    ground_mask = np.abs(residuals) <= ground_threshold
    above_mask = residuals > above_threshold
    below_mask = residuals < below_threshold

    rows, cols, valid, h, w = metric_to_bev_pixels(
        points_roi,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution,
    )

    residuals_valid = residuals[valid]
    ground_valid = ground_mask[valid]
    above_valid = above_mask[valid]
    below_valid = below_mask[valid]

    bev = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw order:
    # ground first, then below, then above
    # BGR colors:
    # ground = green
    # above/object = orange/red
    # below/depression = blue
    ground_rows = rows[ground_valid]
    ground_cols = cols[ground_valid]
    bev[ground_rows, ground_cols] = (60, 220, 60)

    below_rows = rows[below_valid]
    below_cols = cols[below_valid]
    bev[below_rows, below_cols] = (255, 80, 40)

    above_rows = rows[above_valid]
    above_cols = cols[above_valid]
    bev[above_rows, above_cols] = (40, 120, 255)

    # Text
    cv2.putText(
        bev,
        "RANSAC Ground Plane",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        bev,
        "green=ground | orange=above | blue=below",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    stats = {
        "roi_points": int(len(points_roi)),
        "fit_candidates": int(len(fit_points)),
        "ransac_inliers": int(np.sum(inlier_mask)),
        "ground_points": int(np.sum(ground_mask)),
        "above_points": int(np.sum(above_mask)),
        "below_points": int(np.sum(below_mask)),
        "plane": refined_plane,
        "residual_min": float(np.min(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "residual_max": float(np.max(residuals)),
    }

    return bev, stats


def main():
    os.makedirs("../outputs/images", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(1))
    velo = data.get_velo(0)

    cfg = (
        0.0,    # x_min
        50.0,   # x_max
        -25.0,  # y_min
        25.0,   # y_max
        -3.0,   # z_min
        2.0,    # z_max
        0.10,   # resolution
    )

    bev, stats = make_ransac_ground_bev(velo, cfg)

    out_path = "../outputs/images/ransac_ground_single_frame.png"
    cv2.imwrite(out_path, bev)

    print("=== Phase 8.1: RANSAC Ground Plane Single Frame ===")
    print("saved:", out_path)
    print("roi points:", stats["roi_points"])
    print("fit candidates:", stats["fit_candidates"])
    print("ransac inliers:", stats["ransac_inliers"])
    print("ground points:", stats["ground_points"])
    print("above points:", stats["above_points"])
    print("below points:", stats["below_points"])
    print("plane [a,b,c,d]:", stats["plane"])
    print("residual min:", stats["residual_min"])
    print("residual mean:", stats["residual_mean"])
    print("residual max:", stats["residual_max"])


if __name__ == "__main__":
    main()
