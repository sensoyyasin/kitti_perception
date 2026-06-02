import pykitti
import numpy as np

BASEDIR = "/Users/yasinsensoy/datasets/kitti_raw"
DATE = "2011_09_26"
DRIVE = "0019"

# K_cam2 -> camera intrinsic
# T_cam2_velo -> Lidar/Velodyne -> camera2 extrinsic parameters.
'''
Each RGB frame gives cam2 and cam3 images.
cam2 is left color camera
cam3 is right color camera
Velodyne point cloud has N x 4 format.
K_cam2 is cam2 intrinsic matrix
T_cam2_velo transforms Velodyne points into cam2 coordinates.


Intrinsic matrix: 
K =
[ fx  0  cx ]
[  0 fy  cy ]
[  0  0   1 ]

fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.854

3D -> image pixel : u = fx * X / Z + cx, v = fy * Y / Z + cy

Extrinsic matrix : 
T_cam2_velo =
[ R t ]
[ 0 1 ]

R = 3x3 rotation
T = 3x1 translation

Lidar point -> camera coordinates
'''

def main():
    data = pykitti.raw(BASEDIR, DATE, DRIVE)

    print("number of frames: ", len(data.timestamps))
    print("first timestamp : ", data.timestamps[0])
    print("last timestamp: ", data.timestamps[-1])

    img_cam2, img_cam3 = data.get_rgb(0)
    velo = data.get_velo(0)

    print("=== Camera image s===")
    print("cam2: left RGB camera")
    print("cam3: right RGB camera")
    print("cam2 image size:", img_cam2.size)
    print("cam3 image size:", img_cam3.size)

    print("== Velodyne point cloud ===")
    print("velodyne shape:", velo.shape)
    print("velodyne columns: x, y, z, reflectance")
    print("first 5 points: ", velo[:5])
    print("min:", np.min(velo, axis=0))
    print("max:", np.max(velo, axis=0))
    print("mean:", np.mean(velo, axis=0))

    print("== Calibration ===")
    print("calibration cam2 intrinsic:", data.calib.K_cam2)
    print("T_cam2_velo, Velodyne -> cam2 extrinsic: ", data.calib.T_cam2_velo)
    print("T_cam0_velo, Velodyne -> cam0 extrinsic:: ", data.calib.T_cam0_velo)


if __name__ == "__main__":
    main()
