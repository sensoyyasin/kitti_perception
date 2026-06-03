from pathlib import Path
from collections import Counter
import numpy as np


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"


def read_label_file(label_path):
    objects = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) == 0:
            continue

        obj = {
            "type": parts[0],
            "truncation": float(parts[1]),
            "occlusion": int(parts[2]),
            "alpha": float(parts[3]),
            "box": np.array([float(v) for v in parts[4:8]], dtype=np.float32),
            "dimensions_hwl": np.array([float(v) for v in parts[8:11]], dtype=np.float32),
            "location_xyz_camera": np.array([float(v) for v in parts[11:14]], dtype=np.float32),
            "rotation_y": float(parts[14]),
        }

        objects.append(obj)

    return objects


def read_calib_file(calib_path):
    calib = {}

    with open(calib_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        values = np.array([float(x) for x in value.strip().split()], dtype=np.float64)
        calib[key] = values

    return calib


def print_calib_matrix(name, values):
    if len(values) == 12:
        mat = values.reshape(3, 4)
    elif len(values) == 9:
        mat = values.reshape(3, 3)
    else:
        print(name, "shape unknown, len:", len(values))
        return

    print()
    print(name)
    print(mat)


def main():
    root = BASEDIR / SPLIT

    image_dir = root / "image_2"
    velo_dir = root / "velodyne"
    label_dir = root / "label_2"
    calib_dir = root / "calib"

    print("=== KITTI Object Dataset Inspection ===")
    print("root:", root)
    print()

    print("image_2 exists:", image_dir.exists())
    print("velodyne exists:", velo_dir.exists())
    print("label_2 exists:", label_dir.exists())
    print("calib exists:", calib_dir.exists())
    print()

    image_files = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
    velo_files = sorted(velo_dir.glob("*.bin")) if velo_dir.exists() else []
    label_files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    calib_files = sorted(calib_dir.glob("*.txt")) if calib_dir.exists() else []

    print("number of images:", len(image_files))
    print("number of velodyne files:", len(velo_files))
    print("number of label files:", len(label_files))
    print("number of calib files:", len(calib_files))
    print()

    common_ids = sorted(
        set(p.stem for p in image_files)
        & set(p.stem for p in velo_files)
        & set(p.stem for p in label_files)
        & set(p.stem for p in calib_files)
    )

    print("number of complete image/velodyne/label/calib frames:", len(common_ids))

    if len(common_ids) == 0:
        print("No complete frames found.")
        return

    sample_id = common_ids[0]

    print()
    print("sample id:", sample_id)
    print("image:", image_dir / f"{sample_id}.png")
    print("velodyne:", velo_dir / f"{sample_id}.bin")
    print("label:", label_dir / f"{sample_id}.txt")
    print("calib:", calib_dir / f"{sample_id}.txt")
    print()

    class_counter = Counter()
    total_objects = 0

    for frame_id in common_ids:
        objects = read_label_file(label_dir / f"{frame_id}.txt")
        total_objects += len(objects)

        for obj in objects:
            class_counter[obj["type"]] += 1

    print("=== Class distribution over complete subset ===")
    print("total objects:", total_objects)
    for cls, count in class_counter.most_common():
        print(f"{cls}: {count}")

    objects = read_label_file(label_dir / f"{sample_id}.txt")

    print()
    print("=== Sample labels ===")
    print("objects in sample:", len(objects))

    for i, obj in enumerate(objects[:10]):
        print()
        print("object", i)
        print("type:", obj["type"])
        print("2D bbox [left, top, right, bottom]:", obj["bbox"])
        print("3D dimensions [h, w, l]:", obj["dimensions_hwl"])
        print("3D location xyz in camera frame:", obj["location_xyz_camera"])
        print("rotation_y:", obj["rotation_y"])

    calib = read_calib_file(calib_dir / f"{sample_id}.txt")

    print()
    print("=== Calibration keys ===")
    for key, values in calib.items():
        print(key, "len:", len(values))

    if "P2" in calib:
        print_calib_matrix("P2, projection matrix for image_2 / left RGB camera", calib["P2"])

    if "R0_rect" in calib:
        print_calib_matrix("R0_rect, rectification matrix", calib["R0_rect"])

    if "Tr_velo_to_cam" in calib:
        print_calib_matrix("Tr_velo_to_cam, Velodyne to camera transform", calib["Tr_velo_to_cam"])

    print()
    print("note, kitti object labels provide 3D box locations in camera coordinates, not Velodyne coordinates.")
    print("For BEV visualization, we will convert box corners from camera frame back to Velodyne frame.")


if __name__ == "__main__":
    main()
