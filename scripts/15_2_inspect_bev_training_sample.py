from pathlib import Path
import numpy as np
import cv2


BASEDIR = Path("/Users/yasinsensoy/kitti_object")
PROCESSED_DIR = BASEDIR / "processed_bev"

INPUT_DIR = PROCESSED_DIR / "inputs"
MASK_DIR = PROCESSED_DIR / "masks_vehicle"
DEBUG_DIR = PROCESSED_DIR / "debug"

OUTPUT_DIR = Path("../outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_ID = "000613"


CHANNEL_NAMES = [
    "occupancy",
    "density",
    "height",
    "intensity",
]


def normalize_to_uint8(channel):
    channel = np.asarray(channel, dtype=np.float32)
    channel = np.clip(channel, 0.0, 1.0)
    return (channel * 255).astype(np.uint8)


def add_title(image, title):
    out = image.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def make_channel_panel(channel, title, colormap=None):
    gray = normalize_to_uint8(channel)

    if colormap is None:
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        panel = cv2.applyColorMap(gray, colormap)
        panel[gray == 0] = (0, 0, 0)

    panel = add_title(panel, title)

    return panel


def main():
    input_path = INPUT_DIR / f"{FRAME_ID}.npy"
    mask_path = MASK_DIR / f"{FRAME_ID}.png"
    debug_path = DEBUG_DIR / f"{FRAME_ID}.png"

    print("=== 14_10 Inspect BEV Training Sample ===")
    print("frame:", FRAME_ID)
    print("input:", input_path)
    print("mask:", mask_path)
    print("debug:", debug_path)
    print()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    bev = np.load(input_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    print("BEV tensor:")
    print("shape:", bev.shape)
    print("dtype:", bev.dtype)
    print()

    if bev.ndim != 3:
        raise ValueError("BEV input should have shape H x W x C")

    h, w, c = bev.shape

    print("height:", h)
    print("width:", w)
    print("channels:", c)
    print()

    for ch in range(c):
        channel = bev[:, :, ch]
        name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"channel_{ch}"

        print(f"channel {ch} - {name}")
        print("  min:", float(channel.min()))
        print("  mean:", float(channel.mean()))
        print("  max:", float(channel.max()))
        print("  nonzero pixels:", int(np.sum(channel > 0)))
        print()

    print("Mask:")
    print("shape:", mask.shape)
    print("dtype:", mask.dtype)
    print("unique values:", np.unique(mask))
    print("positive pixels:", int(np.sum(mask > 0)))
    print()

    if mask.shape != (h, w):
        print("WARNING: mask shape does not match BEV spatial shape.")
    else:
        print("Shape check: OK, mask matches BEV height/width.")

    # Convert mask to binary 0/1 for sanity stats.
    mask_binary = (mask > 0).astype(np.uint8)

    print()
    print("Binary mask:")
    print("unique values:", np.unique(mask_binary))
    print("positive pixels:", int(np.sum(mask_binary)))

    occupancy = bev[:, :, 0]
    density = bev[:, :, 1]
    height_map = bev[:, :, 2]
    intensity = bev[:, :, 3]

    occ_panel = make_channel_panel(occupancy, "Input channel 0: occupancy")
    density_panel = make_channel_panel(density, "Input channel 1: density")
    height_panel = make_channel_panel(height_map, "Input channel 2: height", cv2.COLORMAP_JET)
    intensity_panel = make_channel_panel(intensity, "Input channel 3: intensity", cv2.COLORMAP_TURBO)

    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_vis, contours, -1, (0, 255, 0), 2)
    mask_vis = add_title(mask_vis, "Target: vehicle mask")

    top = np.hstack([occ_panel, density_panel])
    bottom = np.hstack([height_panel, intensity_panel])

    features_panel = np.vstack([top, bottom])

    mask_panel = cv2.resize(
        mask_vis,
        (features_panel.shape[1], features_panel.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    combined = np.hstack([features_panel, mask_panel])

    out_path = OUTPUT_DIR / f"inspect_bev_training_sample_{FRAME_ID}.png"
    cv2.imwrite(str(out_path), combined)

    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
