import numpy as np
import pykitti

'''
LiDAR point:
[10, 0, 0]

↓ extrinsic: T_cam2_velo

Camera point:
[0.0617, 0.0294, 9.7273]

↓ intrinsic: K_cam2

Image pixel:
[u, v] = [614.14, 175.03]
'''

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"


def main():
    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    R = T_cam2_velo[:3, :3]
    t = T_cam2_velo[:3, 3]

    print("=== Project a single camera point to image pixel ===")
    print("Using cam2 intrinsic K_cam2")
    print("Using extrinsic T_cam2_velo: Velodyne -> cam2")

    # LiDAR point:
    p_velo = np.array([10.0, 0.0, 0.0])

    # LiDAR -> camera coordinates
    p_cam = R @ p_velo + t

    X = p_cam[0]
    Y = p_cam[1]
    Z = p_cam[2]

    print("LiDAR point p_velo: ", p_velo)
    print("Camera point p_cam: ", p_cam)

    print("Parsed camera coordinates:")
    print(f"X = {X}")
    print(f"Y = {Y}")
    print(f"Z = {Z}")

    # camera coordinates -> image pixel
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    print("\nIntrinsic parameters:")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    print("\nProjected image pixel:")
    print(f"u = {u}")
    print(f"v = {v}")

    img_cam2, _ = data.get_rgb(0)
    width, height = img_cam2.size

    print("\nImage size:")
    print(f"width = {width}")
    print(f"height = {height}")

    if Z <= 0:
        print("\nInvalid: point is behind the camera.")
    elif 0 <= u < width and 0 <= v < height:
        print("\nValid: point projects inside the image.")
    else:
        print("\nInvalid: point projects outside the image bounds.")


if __name__ == "__main__":
    main()
