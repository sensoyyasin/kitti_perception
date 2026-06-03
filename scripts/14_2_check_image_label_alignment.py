from pathlib import Path
import cv2
import numpy as np


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"
FRAME_ID = "000000"

OUTPUT_DIR = Path("../outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_COLORS = {
    "Car": (0, 255, 0),
    "Van": (0, 180, 0),
    "Truck": (0, 140, 0),
    "Pedestrian": (255, 0, 0),
    "Person_sitting": (255, 120, 0),
    "Cyclist": (0, 255, 255),
    "Tram": (180, 0, 255),
    "Misc": (180, 180, 180),
    "DontCare": (80, 80, 80),
}


def read_label_file(label_path):
    objects = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            obj = {
                "type": parts[0],
                "truncation": float(parts[1]),
                "occlusion": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": np.array([float(v) for v in parts[4:8]], dtype=np.float32),
                "dimensions_hwl": np.array([float(v) for v in parts[8:11]], dtype=np.float32),
                "location_xyz_camera": np.array([float(v) for v in parts[11:14]], dtype=np.float32),
                "rotation_y": float(parts[14]),
            }

            objects.append(obj)

    return objects


def draw_boxes(image_bgr, objects):
    image = image_bgr.copy()

    for obj in objects:
        cls = obj["type"]
        x1, y1, x2, y2 = obj["bbox"].astype(int)

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        thickness = 2
        if cls == "DontCare":
            thickness = 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"{cls}"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return image


def add_title(image, title):
    out = image.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def main():
    root = BASEDIR / SPLIT

    image_path = root / "image_2" / f"{FRAME_ID}.png"
    label_path = root / "label_2" / f"{FRAME_ID}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    objects = read_label_file(label_path)

    print("frame:", FRAME_ID)
    print("image:", image_path)
    print("label:", label_path)
    print("objects:", len(objects))

    for i, obj in enumerate(objects):
        print(i, obj["type"], "bbox:", obj["bbox"])

    image_raw = add_title(image, f"Raw image_2/{FRAME_ID}.png")
    image_boxes = draw_boxes(image, objects)
    image_boxes = add_title(image_boxes, f"Label overlay label_2/{FRAME_ID}.txt")

    combined = np.hstack([image_raw, image_boxes])

    out_path = OUTPUT_DIR / f"alignment_image_label_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), combined)

    print("saved:", out_path)


if __name__ == "__main__":
    main()
