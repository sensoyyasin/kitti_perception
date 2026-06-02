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

    return velo[mask]


def metric_to_bev_pixels(points, x_min, x_max, y_min, y_max, resolution):
    x = points[:, 0]
    y = points[:, 1]

    rows = ((x_max - x) / resolution).astype(np.int32)
    cols = ((y_max - y) / resolution).astype(np.int32)

    bev_height = int((x_max - x_min) / resolution)
    bev_width = int((y_max - y_min) / resolution)

    valid = (
        (rows >= 0) & (rows < bev_height) &
        (cols >= 0) & (cols < bev_width)
    )

    return rows[valid], cols[valid], valid, bev_height, bev_width


def make_bev_feature_maps(velo,x_min,x_max,y_min,y_max,z_min,z_max,resolution,density_max_count=5):
    velo_roi = filter_roi(velo,x_min,x_max,y_min,y_max,z_min,z_max)

    points_roi = velo_roi[:, :3]
    reflectance_roi = velo_roi[:, 3]

    rows, cols, valid, bev_height, bev_width = metric_to_bev_pixels(
        points_roi,x_min,x_max,y_min,y_max,resolution)

    z_values = points_roi[:, 2][valid]
    reflectance = reflectance_roi[valid]

    occupancy = np.zeros((bev_height, bev_width), dtype=np.uint8)
    occupancy[rows, cols] = 255

    density = np.zeros((bev_height, bev_width), dtype=np.float32)

    for r, c in zip(rows, cols):
        density[r, c] += 1.0

    density_img = np.clip(density, 0, density_max_count)
    density_img = (density_img / density_max_count * 255).astype(np.uint8)

    height = np.full((bev_height, bev_width), -np.inf, dtype=np.float32)

    for r, c, z in zip(rows, cols, z_values):
        if z > height[r, c]:
            height[r, c] = z

    empty_height = height == -np.inf

    height_for_vis = height.copy()
    height_for_vis[empty_height] = z_min

    height_img = (height_for_vis - z_min) / (z_max - z_min)
    height_img = np.clip(height_img, 0.0, 1.0)
    height_img = (height_img * 255).astype(np.uint8)
    height_img[empty_height] = 0

    intensity = np.zeros((bev_height, bev_width), dtype=np.float32)

    for r, c, refl in zip(rows, cols, reflectance):
        if refl > intensity[r, c]:
            intensity[r, c] = refl

    intensity_img = np.clip(intensity, 0.0, 1.0)
    intensity_img = (intensity_img * 255).astype(np.uint8)

    return occupancy, density_img, height_img, intensity_img


def to_color_panel(gray, title, colormap=None):
    if colormap is None:
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        panel = cv2.applyColorMap(gray, colormap)
        panel[gray == 0] = (0, 0, 0)

    cv2.putText(panel,title,(15, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2,cv2.LINE_AA)

    return panel


def main():
    os.makedirs("../outputs/videos", exist_ok=True)

    data = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(NUM_FRAMES))

    x_min, x_max = 0.0, 50.0
    y_min, y_max = -25.0, 25.0
    z_min, z_max = -3.0, 2.0
    resolution = 0.10

    bev_height = int((x_max - x_min) / resolution)
    bev_width = int((y_max - y_min) / resolution)

    output_width = bev_width * 2
    output_height = bev_height * 2

    output_path = "../outputs/videos/bev_feature_maps_video.mp4"
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height),
    )

    for i in range(NUM_FRAMES):
        velo = data.get_velo(i)

        occupancy, density, height, intensity = make_bev_feature_maps(
            velo,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            resolution,
            density_max_count=5,
        )

        occ_panel = to_color_panel(occupancy, "Occupancy")
        den_panel = to_color_panel(density, "Density")
        height_panel = to_color_panel(height, "Height", cv2.COLORMAP_JET)
        intensity_panel = to_color_panel(intensity, "Intensity", cv2.COLORMAP_TURBO)

        top = np.hstack([occ_panel, den_panel])
        bottom = np.hstack([height_panel, intensity_panel])
        frame = np.vstack([top, bottom])

        cv2.putText(
            frame,
            f"KITTI BEV Feature Maps | Frame {i}",
            (20, output_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if i % 25 == 0:
            print(f"processed frame {i}/{NUM_FRAMES}")

    writer.release()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
