from pathlib import Path
from collections import Counter


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
SPLIT = "training"


def read_label_classes(label_path):
    counts = Counter()

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            cls = parts[0]

            if cls == "DontCare":
                continue

            counts[cls] += 1

    return counts


def main():
    root = BASEDIR / SPLIT

    image_dir = root / "image_2"
    velo_dir = root / "velodyne"
    label_dir = root / "label_2"
    calib_dir = root / "calib"

    image_ids = set(p.stem for p in image_dir.glob("*.png"))
    velo_ids = set(p.stem for p in velo_dir.glob("*.bin"))
    label_ids = set(p.stem for p in label_dir.glob("*.txt"))
    calib_ids = set(p.stem for p in calib_dir.glob("*.txt"))

    common_ids = sorted(image_ids & velo_ids & label_ids & calib_ids)

    print("complete frames:", len(common_ids))
    print()

    scored = []

    for frame_id in common_ids:
        label_path = label_dir / f"{frame_id}.txt"
        counts = read_label_classes(label_path)

        car_like = counts["Car"] + counts["Van"] + counts["Truck"]
        people_like = counts["Pedestrian"] + counts["Person_sitting"]
        cyclist_like = counts["Cyclist"]

        score = (
            car_like * 3
            + cyclist_like * 3
            + people_like * 2
            + counts["Tram"] * 2
            + counts["Misc"]
        )

        total = sum(counts.values())

        if total > 0:
            scored.append((score, total, frame_id, counts))

    scored.sort(reverse=True)

    print("Top interesting frames:")
    for score, total, frame_id, counts in scored[:30]:
        print(
            frame_id,
            "score:", score,
            "total:", total,
            "classes:", dict(counts),
        )


if __name__ == "__main__":
    main()
