'''
Small U-Net best checkpoint download
Read the test split
Use threshold = 0.70
Calculate IoU, F1, precision, recall
Save prediction images
'''

from pathlib import Path
import sys
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "15_3_train_bev_vehicle_segmentation.py"

spec = importlib.util.spec_from_file_location("train_bev", TRAIN_SCRIPT)
train_bev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_bev)


MODEL_TYPE = "small_unet"
THRESHOLD = 0.70
BATCH_SIZE = 2

BASE_DIR = Path("/Users/yasinsensoy/kitti_object")
PROCESSED_DIR = BASE_DIR / "processed_bev"

INPUT_DIR = PROCESSED_DIR / "inputs"
MASK_DIR = PROCESSED_DIR / "masks_vehicle"

EXPERIMENT_NAME = "bev_vehicle_small_unet_448x320_e20_pw8p0"

MODEL_DIR = PROCESSED_DIR / "models" / EXPERIMENT_NAME
CHECKPOINT_PATH = MODEL_DIR / "bev_vehicle_segmentation_best.pth"
VAL_IDS_PATH = MODEL_DIR / "splits" / "val_ids.txt"

OUT_DIR = PROCESSED_DIR / "predictions" / f"{EXPERIMENT_NAME}_final_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUT_DIR / "final_val_metrics.csv"

def save_prediction_debug(bev_tensor, target_tensor, prob_tensor, frame_id, threshold):

    bev_img = train_bev.tensor_to_image(bev_tensor.cpu())

    gt = target_tensor.squeeze().cpu().numpy()
    prob = prob_tensor.squeeze().cpu().numpy()
    pred = (prob > threshold).astype(np.uint8)

    gt_img = (gt * 255).astype(np.uint8)
    pred_img = (pred * 255).astype(np.uint8)

    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

    overlay = bev_img.copy()

    overlay[gt > 0] = [0, 0, 255]

    overlay[pred > 0] = [0, 255, 0]

    overlap = (gt > 0) & (pred > 0)
    overlay[overlap] = [0, 255, 255]

    combined = np.hstack([bev_img, gt_bgr, pred_bgr, overlay])

    cv2.imwrite(str(OUT_DIR / f"{frame_id}_eval.png"), combined)

def main():
    print("Loading Small U-Net final evaluation...")

    device = train_bev.get_device()
    print("device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    if not VAL_IDS_PATH.exists():
        raise FileNotFoundError(f"Missing val ids: {VAL_IDS_PATH}")

    val_ids = train_bev.read_ids(VAL_IDS_PATH)
    print("number of validation samples:", len(val_ids))

    dataset = train_bev.BEVVehicleDataset(
        frame_ids=val_ids,
        input_dir=INPUT_DIR,
        mask_dir=MASK_DIR,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = train_bev.build_model(MODEL_TYPE).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    criterion = train_bev.BCEDiceLoss(pos_weight=8.0).to(device)

    metrics = train_bev.evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        threshold=THRESHOLD,
    )

    print("\nFinal validation evaluation")
    print("model:", MODEL_TYPE)
    print("checkpoint:", CHECKPOINT_PATH)
    print("threshold:", THRESHOLD)
    print("loss:", metrics["loss"])
    print("iou:", metrics["iou"])
    print("f1:", metrics["f1"])
    print("precision:", metrics["precision"])
    print("recall:", metrics["recall"])

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "checkpoint", "threshold", "loss", "iou", "f1", "precision", "recall"])
        writer.writerow([
            MODEL_TYPE,
            str(CHECKPOINT_PATH),
            THRESHOLD,
            metrics["loss"],
            metrics["iou"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        ])

    print("\nSaved metrics to:", METRICS_CSV)

    print("\nSaving prediction debug images...")

    model.eval()
    saved = 0
    max_save = 20

    with torch.no_grad():
        for x, y, frame_ids in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            for i in range(x.shape[0]):
                if saved >= max_save:
                    break

                save_prediction_debug(
                    bev_tensor=x[i],
                    target_tensor=y[i],
                    prob_tensor=probs[i],
                    frame_id=frame_ids[i],
                    threshold=THRESHOLD,
                )

                saved += 1

            if saved >= max_save:
                break

    print("Saved prediction images to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
