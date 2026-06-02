import numpy as np
import pykitti

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"

def print_matrix(name, matrix):
    print(f"\n{name}:")
    print(matrix)

def main():
    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    print("===Camera-LiDAR Calibration Inspection===")
    print("Camera used: cam2, left RGB camera")
    print("Transform used: T_cam2_velo, Velodyne -> cam2")

    # Intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print_matrix("K_cam2 intrinsic matrix", K)

    print("\nIntrinsic parameters:")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    print("\nIntrinsic:")
    print("fx, fy: focal length in pixel units")
    print("cx, cy: principal point / optical center in pixel coordinates")

    R = T_cam2_velo[:3, :3]
    t = T_cam2_velo[:3, 3]

    print_matrix("T_cam2_velo extrinsic matrix, Velodyne -> cam2", T_cam2_velo)
    print_matrix("R, rotation part", R)

    print("\nt, translation part:")
    print(t)

    print("\nExtrinsic meaning:")
    print("p_cam2 = R @ p_velo + t")
    print("or using homogeneous coordinates:")
    print("p_cam2_h = T_cam2_velo @ p_velo_h")

    identity_check = R.T @ R
    determinant = np.linalg.det(R)

    print_matrix("R.T @ R, should be close to identity", identity_check)
    print(f"\ndet(R), should be close to 1: {determinant}")

    img_cam2, _ = data.get_rgb(0)
    width, height = img_cam2.size

    print("\nImage size:")
    print(f"width = {width}")
    print(f"height = {height}")

    print("\nPrincipal point compared to image center:")
    print(f"image center x = {width / 2}")
    print(f"image center y = {height / 2}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")


if __name__ == "__main__":
    main()
