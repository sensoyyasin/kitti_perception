"""
16  Monte Carlo: how many vehicles can LiDAR still "see" in fog?

What we simulate:
    "If this exact KITTI scene had been recorded in fog, how many LiDAR returns
    would each car still have produced?"
Fog is random at the level of single light pulses, so the same scene in the
same fog gives a slightly different scan every time. Monte Carlo = simulate it
many times and average.

Steps in this file:
    1. read one KITTI frame (points, labels, calibration)
    2. count the LiDAR points inside each vehicle box
    3. fog: give every point a chance to survive, then roll the dice
    4. Monte Carlo loop: all frames x fog levels x several random seeds
    5. summary table: % of vehicles still visible, per distance
    6. OpenCV windows: one scene from above in each fog level, and the charts
       (nothing is saved to disk; press any key to close)

Run:
    python 16_monte_carlo_fog.py
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path.home() / "datasets" / "kitti_object" / "training"

# Fog density alpha (1/m), the "extinction coefficient": the fraction of light lost per meter.
# Where the values come from:
#   - Visibility in fog is measured as the Meteorological Optical Range (MOR): the distance
#     at which light drops to 5% of its strength. From exp(-alpha * MOR) = 0.05 we get
#     MOR = ln(20) / alpha ~ 3 / alpha.
#   - Hahner et al., "Fog Simulation on Real LiDAR Point Clouds", ICCV 2021, use
#     alpha in [0, 0.005, 0.01, 0.02, 0.03, 0.06] for MOR ~ [inf, 600, 300, 150, 100, 50] m.
#     We take three of their levels:
#         alpha = 0.01  ->  ~300 m visibility  (light fog)
#         alpha = 0.03  ->  ~100 m visibility  (fog)
#         alpha = 0.06  ->   ~50 m visibility  (dense fog)
#   - Official definition: fog = visibility below 1 km, so all three are real fog.
FOG_LEVELS = [0.01, 0.03, 0.06]

N_SEEDS = 5
MIN_POINTS = 10
DISTANCE_BINS = [(0, 20), (20, 40), (40, 70)]
VEHICLES = {"Car", "Van", "Truck"}

EXAMPLE_FRAME = "000008"            # scene shown in the OpenCV window

def read_frame(frame_id):
    points = np.fromfile(DATA_DIR / "velodyne" / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)

    calib = {}
    for line in open(DATA_DIR / "calib" / f"{frame_id}.txt"):
        if ":" in line:
            key, values = line.split(":", 1)
            calib[key] = np.array(values.split(), dtype=np.float64)

    vehicles = []
    for line in open(DATA_DIR / "label_2" / f"{frame_id}.txt"):
        p = line.split()
        if p and p[0] in VEHICLES:
            vehicles.append({
                "hwl": np.array(p[8:11], dtype=np.float64),
                "center_cam": np.array(p[11:14], dtype=np.float64),
                "rotation_y": float(p[14]),
            })
    return points, calib, vehicles


def lidar_to_camera_matrix(calib):
    R0 = np.eye(4)
    R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)
    return R0 @ Tr


def lidar_to_camera(points_xyz, calib):
    """KITTI labels live in the camera frame, so we move the LiDAR points there."""
    T = lidar_to_camera_matrix(calib)
    return points_xyz @ T[:3, :3].T + T[:3, 3]


def box_footprint_lidar(vehicle, calib):
    """The 4 bottom corners of a vehicle box, in LiDAR coordinates (for drawing from above)."""
    h, w, l = vehicle["hwl"]
    c, s = np.cos(vehicle["rotation_y"]), np.sin(vehicle["rotation_y"])
    local = np.array([[l / 2, 0, w / 2], [l / 2, 0, -w / 2], [-l / 2, 0, -w / 2], [-l / 2, 0, w / 2]])
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners_cam = local @ R.T + vehicle["center_cam"]
    T_inv = np.linalg.inv(lidar_to_camera_matrix(calib))
    return corners_cam @ T_inv[:3, :3].T + T_inv[:3, 3]



def inside_box(points_cam, vehicle):
    """True for every point that falls inside the vehicle's 3D box."""
    h, w, l = vehicle["hwl"]
    c, s = np.cos(vehicle["rotation_y"]), np.sin(vehicle["rotation_y"])
    d = points_cam - vehicle["center_cam"]
    x = c * d[:, 0] - s * d[:, 2]          # rotate into the box's own frame
    z = s * d[:, 0] + c * d[:, 2]
    y = d[:, 1]
    return (np.abs(x) <= l / 2) & (np.abs(z) <= w / 2) & (y >= -h) & (y <= 0)


def count_points_in_box(points_cam, vehicle):
    return int(inside_box(points_cam, vehicle).sum())


def apply_fog(points, alpha, rng):
    """
    Fog absorbs light. A LiDAR pulse travels to the object and back (2 x distance),
    so the chance that a point survives is exp(-2 * alpha * distance)
    (Beer-Lambert law). Far points are lost first.

    Then we roll one die per point: this is the random part of Monte Carlo.
    """
    distance = np.linalg.norm(points[:, :3], axis=1)
    survive_probability = np.exp(-2 * alpha * distance)
    survives = rng.random(len(points)) < survive_probability
    return points[survives]


def run():
    frame_ids = sorted(p.stem for p in (DATA_DIR / "velodyne").glob("*.bin"))
    print(f"frames: {len(frame_ids)}, fog levels: {FOG_LEVELS}, seeds: {N_SEEDS}\n")

    results = []

    for frame_id in frame_ids:
        points, calib, vehicles = read_frame(frame_id)
        if not vehicles:
            continue

        distances = [float(np.hypot(v["center_cam"][0], v["center_cam"][2])) for v in vehicles]

        clean_cam = lidar_to_camera(points[:, :3], calib)
        clean_counts = [count_points_in_box(clean_cam, v) for v in vehicles]

        for alpha in FOG_LEVELS:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng([seed, int(frame_id)])
                foggy = apply_fog(points, alpha, rng)
                foggy_cam = lidar_to_camera(foggy[:, :3], calib)

                for v, dist, n_clean in zip(vehicles, distances, clean_counts):
                    if n_clean < MIN_POINTS:
                        continue
                    n_foggy = count_points_in_box(foggy_cam, v)
                    results.append((alpha, dist, n_clean, n_foggy))

    return np.array(results)


def visible_rates(results):
    """% of vehicles that still have >= MIN_POINTS points, for every (fog level, distance bin)."""
    alpha, dist, _, n_foggy = results.T
    visible = n_foggy >= MIN_POINTS
    return {(a, b): visible[(alpha == a) & (dist >= b[0]) & (dist < b[1])].mean()
            for a in FOG_LEVELS for b in DISTANCE_BINS}


def summarize(results):
    rates = visible_rates(results)
    header = "fog alpha  " + "".join(f"{f'{a}-{b} m':>12}" for a, b in DISTANCE_BINS)
    print(header)
    print("-" * len(header))
    for a in FOG_LEVELS:
        print(f"{a:<11}" + "".join(f"{rates[(a, b)]:>11.0%} " for b in DISTANCE_BINS))
    print(f"\n% of vehicles that still have >= {MIN_POINTS} LiDAR points")


# OpenCV colors are BGR
WHITE, GRAY = (255, 255, 255), (150, 150, 150)
BLUE, RED = (230, 140, 40), (60, 60, 230)
RESOLUTION = 0.1
X_MAX, Y_HALF = 50.0, 20.0


def to_pixel(x, y):
    """LiDAR (x forward, y left) -> image (row, col): forward is up, left is left."""
    row = (X_MAX - x) / RESOLUTION
    col = (Y_HALF - y) / RESOLUTION
    return row, col


def draw_top_view(points, calib, vehicles, title):
    h, w = int(X_MAX / RESOLUTION), int(2 * Y_HALF / RESOLUTION)
    img = np.zeros((h + 60, w, 3), np.uint8)

    keep = (points[:, 0] > 0) & (points[:, 0] < X_MAX) & (np.abs(points[:, 1]) < Y_HALF) & (points[:, 2] < 0.5)
    rows, cols = to_pixel(points[keep, 0], points[keep, 1])
    img[rows.astype(int) + 60, cols.astype(int)] = GRAY

    points_cam = lidar_to_camera(points[:, :3], calib)
    n_visible = n_shown = 0
    for v in vehicles:
        corners = box_footprint_lidar(v, calib)
        if corners[:, 0].mean() > X_MAX:
            continue
        n = count_points_in_box(points_cam, v)
        visible = n >= MIN_POINTS
        n_visible += visible
        n_shown += 1
        r, c = to_pixel(corners[:, 0], corners[:, 1])
        poly = np.stack([c, r + 60], axis=1).astype(np.int32)
        color = BLUE if visible else RED
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)
        cv2.putText(img, str(n), (int(c.max()) + 4, int(r.mean()) + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"{keep.sum():,} points, {n_visible}/{n_shown} cars visible", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, img.shape[0] - 6), 5, (0, 255, 255), -1)
    return img


def show_scene_in_fog(frame_id=EXAMPLE_FRAME):
    """One scene from above: clean, then one roll of the dice for each fog level."""
    points, calib, vehicles = read_frame(frame_id)
    panels = [draw_top_view(points, calib, vehicles, f"{frame_id} clean")]
    for alpha in FOG_LEVELS:
        foggy = apply_fog(points, alpha, np.random.default_rng(0))
        panels.append(draw_top_view(foggy, calib, vehicles, f"alpha={alpha} (~{3 / alpha:.0f} m vis.)"))
    sep = np.full((panels[0].shape[0], 4, 3), 80, np.uint8)
    row = np.hstack([p if i == 0 else np.hstack([sep, p]) for i, p in enumerate(panels)])
    legend = np.zeros((34, row.shape[1], 3), np.uint8)
    cv2.putText(legend, f"blue box: car still has >= {MIN_POINTS} points   red box: lost in fog   "
                        f"number: points on the car   yellow dot: our LiDAR",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imshow("Fog from above", np.vstack([row, legend]))


def show_charts(results):
    """Fog formula and Monte Carlo result, drawn with matplotlib and shown with OpenCV."""
    colors = ["#6da7ec", "#2a78d6", "#104281"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    d = np.linspace(0, 70, 200)
    for a, color in zip(FOG_LEVELS, colors):
        ax1.plot(d, np.exp(-2 * a * d), color=color, linewidth=2, label=f"alpha = {a}")
    ax1.set_title("Chance one point survives: exp(-2 x alpha x distance)", loc="left", fontsize=11)
    ax1.set_xlabel("distance (m)")
    ax1.legend(frameon=False)

    rates = visible_rates(results)
    x = np.arange(len(DISTANCE_BINS))
    for i, (a, color) in enumerate(zip(FOG_LEVELS, colors)):
        ax2.bar(x + (i - 1) * 0.26, [rates[(a, b)] for b in DISTANCE_BINS], 0.23,
                color=color, label=f"alpha = {a}")
    ax2.set_xticks(x, [f"{lo}-{hi} m" for lo, hi in DISTANCE_BINS])
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_title(f"Monte Carlo: cars still with >= {MIN_POINTS} points", loc="left", fontsize=11)
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    cv2.imshow("Fog Monte Carlo", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))


if __name__ == "__main__":
    results = run()
    summarize(results)
    show_scene_in_fog()
    show_charts(results)
    print("\nPress any key in an OpenCV window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
