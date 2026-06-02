import numpy as np
import pykitti

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"

def main():
    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    T_cam2_velo = data.calib.T_cam2_velo

    R = T_cam2_velo[:3, :3]
    t = T_cam2_velo[:3, 3]

    print("===Transform a single LiDAR point to camera coordinates===")
    print("Using T_cam2_velo: Velodyne -> cam2")

    print("R: ", R)
    print("t: ", t)

    # A simple LiDAR point, KITTI Velodyne coordinates : x = forward, y = left, z = up
    p_velo = np.array([10.0, 0.0, 0.0])

    print("\nLiDAR point p_velo: ", p_velo)
    print("Meaning: 10 meters forward from the LiDAR, centered left/right, same height as LiDAR origin.")

    # Method 1: R @ p + t
    p_cam_manual = R @ p_velo + t

    print("\nCamera point using R @ p_velo + t:")
    print(p_cam_manual)

    # Method 2: homogeneous transform
    p_velo_h = np.array([10.0, 0.0, 0.0, 1.0])
    p_cam_h = T_cam2_velo @ p_velo_h

    print("\nLiDAR point homogeneous p_velo_h:")
    print(p_velo_h)

    print("\nCamera point homogeneous using T_cam2_velo @ p_velo_h:")
    print(p_cam_h)

    print("\nCompare:")
    print("manual 3D:", p_cam_manual)
    print("homogeneous first 3 values:", p_cam_h[:3])

    print("\nCamera coordinate interpretation:")
    print("X_cam = left/right in camera frame")
    print("Y_cam = up/down in camera frame")
    print("Z_cam = depth, forward direction of the camera")

    X, Y, Z = p_cam_manual

    print("\nParsed camera coordinates:")
    print(f"X_cam = {X}")
    print(f"Y_cam = {Y}")
    print(f"Z_cam = {Z}")

    if Z > 0:
        print("\nThis point is in front of the camera because Z_cam > 0.")
    else:
        print("\nThis point is behind the camera because Z_cam <= 0.")


if __name__ == "__main__":
    main()
