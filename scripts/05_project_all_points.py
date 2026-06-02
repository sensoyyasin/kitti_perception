import numpy as np
import pykitti

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"

# convert nx3 points to nx4 homogenous coordinates.
# [x, y, z] -> [x, y, z, 1]
def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1))
    return np.hstack([points_xyz, ones])


def main():
    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    K = data.calib.K_cam2
    T_cam2_velo = data.calib.T_cam2_velo

    img_cam2, _ = data.get_rgb(0)
    width, height = img_cam2.size

    velo = data.get_velo(0)
    points_velo = velo[:, :3]

    print("=== Project all LiDAR points to image pixels ===")
    print("Original Velodyne shape:", velo.shape)
    print("points_velo shape:", points_velo.shape)

    # homogeneous coordinates
    points_velo_h = to_homogeneous(points_velo)
    print("points_velo_h shape:", points_velo_h.shape)

    # LiDAR -> camera
    points_cam_h = (T_cam2_velo @ points_velo_h.T).T
    points_cam = points_cam_h[:, :3]

    print("\npoints_cam shape:", points_cam.shape)

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    # We're keeping only points in front of camera
    in_front = Z > 0

    print("points in front of camera:", np.count_nonzero(in_front))
    print("points behind camera:", len(Z) - np.count_nonzero(in_front))

    X = X[in_front]
    Y = Y[in_front]
    Z = Z[in_front]

    # camera coordinates -> image pixels
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # keep only pixels inside image
    inside_image = ( (u >= 0) & (u < width) & (v >= 0) & (v < height) )

    u_valid = u[inside_image]
    v_valid = v[inside_image]
    depth_valid = Z[inside_image]

    print("\nimage width:", width)
    print("image height:", height)
    print("projected points inside image:", len(u_valid))
    print("projected points outside image:", len(u) - len(u_valid))

    print("\nFirst 10 projected points:")
    for i in range(min(10, len(u_valid))):
        print(
            f"pixel=({u_valid[i]:.2f}, {v_valid[i]:.2f}), "
            f"depth={depth_valid[i]:.2f} m"
        )

    print("Depth stats for valid projected points:")
    print("min depth:", np.min(depth_valid))
    print("max depth:", np.max(depth_valid))
    print("mean depth:", np.mean(depth_valid))


if __name__ == "__main__":
    main()
