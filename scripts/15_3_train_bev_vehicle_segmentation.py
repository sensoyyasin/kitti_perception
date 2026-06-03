"""
BEV Vehicle Segmentation Baselines

Check PyTorch / Mac GPU:
    python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"

Task:
    input  = BEV tensor from LiDAR: occupancy, density, height, intensity
    target = BEV vehicle mask

This is segmentation, not full 3D detection.
It predicts vehicle area mask in BEV, not 3D bounding boxes.

Baselines:
    1. Plain 2D CNN
    2. Small U-Net

Metrics:
    IoU
    F1
    Precision
    Recall

Main improvements:
    - fixed train/val split
    - validation threshold search
    - BCE + Dice loss
    - ReduceLROnPlateau scheduler
    - prediction overlay debug images
    - history CSV logging

Outputs:
    /Users/yasinsensoy/kitti_object/processed_bev/models/<experiment_name>/
        bev_vehicle_segmentation_best.pth
        bev_vehicle_segmentation_last.pth
        history.csv
        train_ids.txt
        val_ids.txt

    /Users/yasinsensoy/kitti_object/processed_bev/predictions/<experiment_name>/
        epoch_XXX_<frame_id>.png
"""

from pathlib import Path
import csv
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================

SEED = 42

MODEL_TYPE = "small_unet"  # "plain_cnn" or "small_unet"

BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_RATIO = 0.20

# Original processed BEV is 700 x 500.
# Resize for faster training. Try 512 x 384 later for better quality.
TRAIN_HEIGHT = 448
TRAIN_WIDTH = 320

POS_WEIGHT = 8.0

DEFAULT_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = [
    0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90
]

NUM_WORKERS = 0
DEBUG_SAMPLES_PER_EPOCH = 4
PROGRESS_EVERY_N_BATCHES = 50

USE_SCHEDULER = True
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

EXPERIMENT_NAME = (
    f"bev_vehicle_{MODEL_TYPE}"
    f"_{TRAIN_HEIGHT}x{TRAIN_WIDTH}"
    f"_e{NUM_EPOCHS}"
    f"_pw{str(POS_WEIGHT).replace('.', 'p')}"
)


# =========================
# Paths
# =========================

BASEDIR = Path("/Users/yasinsensoy/kitti_object")
PROCESSED_DIR = BASEDIR / "processed_bev"

INPUT_DIR = PROCESSED_DIR / "inputs"
MASK_DIR = PROCESSED_DIR / "masks_vehicle"

MODEL_DIR = PROCESSED_DIR / "models" / EXPERIMENT_NAME
PRED_DIR = PROCESSED_DIR / "predictions" / EXPERIMENT_NAME
SPLIT_DIR = MODEL_DIR / "splits"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IDS_PATH = SPLIT_DIR / "train_ids.txt"
VAL_IDS_PATH = SPLIT_DIR / "val_ids.txt"
HISTORY_PATH = MODEL_DIR / "history.csv"


# =========================
# Reproducibility
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# Device
# =========================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# Split helpers
# =========================

def read_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def write_ids(path, ids):
    with open(path, "w") as f:
        for frame_id in ids:
            f.write(f"{frame_id}\n")


def get_or_create_split(frame_ids):
    if TRAIN_IDS_PATH.exists() and VAL_IDS_PATH.exists():
        train_ids = read_ids(TRAIN_IDS_PATH)
        val_ids = read_ids(VAL_IDS_PATH)

        # Keep only files that still exist.
        available = set(frame_ids)
        train_ids = [fid for fid in train_ids if fid in available]
        val_ids = [fid for fid in val_ids if fid in available]

        if len(train_ids) > 0 and len(val_ids) > 0:
            print("using existing split files", flush=True)
            return train_ids, val_ids

    print("creating new fixed train/val split", flush=True)

    frame_ids = list(frame_ids)
    random.shuffle(frame_ids)

    val_count = int(len(frame_ids) * VAL_RATIO)
    val_ids = frame_ids[:val_count]
    train_ids = frame_ids[val_count:]

    write_ids(TRAIN_IDS_PATH, train_ids)
    write_ids(VAL_IDS_PATH, val_ids)

    return train_ids, val_ids


# =========================
# Dataset
# =========================

class BEVVehicleDataset(Dataset):
    def __init__(self, frame_ids, input_dir, mask_dir, augment=False):
        self.frame_ids = frame_ids
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.augment = augment

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]

        input_path = self.input_dir / f"{frame_id}.npy"
        mask_path = self.mask_dir / f"{frame_id}.png"

        bev = np.load(input_path).astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        mask = (mask > 0).astype(np.float32)

        bev_resized = np.zeros(
            (TRAIN_HEIGHT, TRAIN_WIDTH, bev.shape[2]),
            dtype=np.float32,
        )

        for c in range(bev.shape[2]):
            bev_resized[:, :, c] = cv2.resize(
                bev[:, :, c],
                (TRAIN_WIDTH, TRAIN_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

        mask_resized = cv2.resize(
            mask,
            (TRAIN_WIDTH, TRAIN_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        if self.augment:
            # BEV left/right flip.
            if random.random() < 0.5:
                bev_resized = np.flip(bev_resized, axis=1).copy()
                mask_resized = np.flip(mask_resized, axis=1).copy()

            # Mild input noise. Do not touch mask.
            if random.random() < 0.25:
                noise = np.random.normal(0.0, 0.015, size=bev_resized.shape).astype(np.float32)
                bev_resized = np.clip(bev_resized + noise, 0.0, 1.0)

        # HWC -> CHW
        bev_tensor = torch.from_numpy(bev_resized).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)

        return bev_tensor, mask_tensor, frame_id


# =========================
# Models
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PlainCNN(nn.Module):
    """
    Simple fully-convolutional baseline.

    No encoder-decoder.
    No skip connections.
    No pooling.

    Input:  B x 4 x H x W
    Output: B x 1 x H x W logits
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """
    Small encoder-decoder U-Net baseline.

    Encoder learns wider context.
    Decoder restores spatial resolution.
    Skip connections preserve local details.

    Input:  B x 4 x H x W
    Output: B x 1 x H x W logits
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out_conv(d1)


def build_model(model_type):
    if model_type == "plain_cnn":
        return PlainCNN(in_channels=4, out_channels=1)

    if model_type == "small_unet":
        return SmallUNet(in_channels=4, out_channels=1)

    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


# =========================
# Loss and metrics
# =========================

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=8.0):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        self.bce.pos_weight = self.pos_weight.to(logits.device)

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return bce_loss + dice_loss


def compute_metrics_from_probs(probs, targets, threshold):
    preds = (probs > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()

    eps = 1e-6

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "f1": float(f1.detach().cpu()),
    }


def compute_metrics(logits, targets, threshold):
    probs = torch.sigmoid(logits)
    return compute_metrics_from_probs(probs, targets, threshold)


def evaluate_model(model, loader, criterion, device, threshold):
    model.eval()

    loss_sum = 0.0
    metrics_sum = {
        "precision": 0.0,
        "recall": 0.0,
        "iou": 0.0,
        "f1": 0.0,
    }

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.detach().cpu())

            metrics = compute_metrics(logits, y, threshold)

            for key in metrics_sum:
                metrics_sum[key] += metrics[key]

    num_batches = len(loader)

    return {
        "loss": loss_sum / num_batches,
        "precision": metrics_sum["precision"] / num_batches,
        "recall": metrics_sum["recall"] / num_batches,
        "iou": metrics_sum["iou"] / num_batches,
        "f1": metrics_sum["f1"] / num_batches,
    }


def find_best_threshold(model, loader, device, thresholds):
    model.eval()

    totals = {
        t: {
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
            "f1": 0.0,
        }
        for t in thresholds
    }

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            for threshold in thresholds:
                metrics = compute_metrics_from_probs(probs, y, threshold)
                for key in totals[threshold]:
                    totals[threshold][key] += metrics[key]

    num_batches = len(loader)

    results = {}
    for threshold in thresholds:
        results[threshold] = {
            key: value / num_batches
            for key, value in totals[threshold].items()
        }

    best_threshold = max(results.keys(), key=lambda t: results[t]["iou"])
    return best_threshold, results


# =========================
# Visualization
# =========================

def tensor_to_image(bev_tensor):
    """
    bev_tensor: 4 x H x W

    RGB visualization:
      R = height
      G = density
      B = occupancy
    """
    bev = bev_tensor.detach().cpu().numpy()

    occupancy = bev[0]
    density = bev[1]
    height = bev[2]

    rgb = np.stack([height, density, occupancy], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def make_overlay_panel(target, pred):
    """
    target and pred are uint8 0/255 images.
    Green = target
    Red   = prediction
    Yellow = overlap
    """
    overlay = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)

    target_mask = target > 0
    pred_mask = pred > 0
    overlap = target_mask & pred_mask

    # BGR format for OpenCV:
    overlay[target_mask] = (0, 180, 0)
    overlay[pred_mask] = (0, 0, 220)
    overlay[overlap] = (0, 220, 220)

    return overlay


def add_title(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def save_prediction_debug(model, dataset, device, epoch, model_type, threshold, max_samples):
    model.eval()

    count = min(max_samples, len(dataset))

    with torch.no_grad():
        for i in range(count):
            x, y, frame_id = dataset[i]

            x_batch = x.unsqueeze(0).to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            pred = (probs > threshold).astype(np.uint8) * 255
            target = y[0].numpy().astype(np.uint8) * 255

            input_vis = tensor_to_image(x)
            input_bgr = cv2.cvtColor(input_vis, cv2.COLOR_RGB2BGR)

            target_bgr = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

            target_bgr[:, :, 1] = np.maximum(target_bgr[:, :, 1], target)
            pred_bgr[:, :, 2] = np.maximum(pred_bgr[:, :, 2], pred)

            overlay = make_overlay_panel(target, pred)

            input_bgr = add_title(input_bgr, "input BEV")
            target_bgr = add_title(target_bgr, "target GT")
            pred_bgr = add_title(pred_bgr, f"prediction thr={threshold:.2f}")
            overlay = add_title(overlay, "overlay: green GT, red pred, yellow overlap")

            combined = np.hstack([input_bgr, target_bgr, pred_bgr, overlay])

            footer = (
                f"{model_type} | frame {frame_id} | epoch {epoch} | "
                f"threshold={threshold:.2f}"
            )

            cv2.rectangle(
                combined,
                (0, combined.shape[0] - 32),
                (combined.shape[1], combined.shape[0]),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                combined,
                footer,
                (12, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            out_path = PRED_DIR / f"epoch_{epoch:03d}_{frame_id}.png"
            cv2.imwrite(str(out_path), combined)


# =========================
# History logging
# =========================

def init_history_csv(path):
    if path.exists():
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_iou",
            "train_f1",
            "train_precision",
            "train_recall",
            "val_loss",
            "val_iou_default",
            "val_f1_default",
            "val_precision_default",
            "val_recall_default",
            "best_threshold",
            "val_iou_best_threshold",
            "val_f1_best_threshold",
            "val_precision_best_threshold",
            "val_recall_best_threshold",
        ])


def append_history_csv(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================
# Main
# =========================

def main():
    device = get_device()

    print("=== 15_1 Train BEV Vehicle Segmentation ===", flush=True)
    print("device:", device, flush=True)
    print("model type:", MODEL_TYPE, flush=True)
    print("experiment:", EXPERIMENT_NAME, flush=True)
    print("input dir:", INPUT_DIR, flush=True)
    print("mask dir:", MASK_DIR, flush=True)
    print("model dir:", MODEL_DIR, flush=True)
    print("prediction dir:", PRED_DIR, flush=True)
    print("", flush=True)

    input_files = sorted(INPUT_DIR.glob("*.npy"))
    frame_ids = [p.stem for p in input_files]
    frame_ids = [fid for fid in frame_ids if (MASK_DIR / f"{fid}.png").exists()]

    print("total samples:", len(frame_ids), flush=True)

    if len(frame_ids) == 0:
        raise RuntimeError("No BEV training samples found.")

    train_ids, val_ids = get_or_create_split(frame_ids)

    print("train samples:", len(train_ids), flush=True)
    print("val samples:", len(val_ids), flush=True)
    print("train size:", TRAIN_HEIGHT, TRAIN_WIDTH, flush=True)
    print("default threshold:", DEFAULT_THRESHOLD, flush=True)
    print("threshold candidates:", THRESHOLD_CANDIDATES, flush=True)
    print("", flush=True)

    train_dataset = BEVVehicleDataset(
        train_ids,
        INPUT_DIR,
        MASK_DIR,
        augment=True,
    )

    val_dataset = BEVVehicleDataset(
        val_ids,
        INPUT_DIR,
        MASK_DIR,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model(MODEL_TYPE).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", num_params, flush=True)
    print("", flush=True)

    criterion = BCEDiceLoss(pos_weight=POS_WEIGHT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
    else:
        scheduler = None

    init_history_csv(HISTORY_PATH)

    best_val_iou = -1.0
    best_threshold_overall = DEFAULT_THRESHOLD

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nstarting epoch {epoch}/{NUM_EPOCHS}", flush=True)

        model.train()

        train_loss_sum = 0.0
        train_metrics_sum = {
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
            "f1": 0.0,
        }

        for batch_idx, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu())

            metrics = compute_metrics(logits, y, DEFAULT_THRESHOLD)
            for key in train_metrics_sum:
                train_metrics_sum[key] += metrics[key]

            if batch_idx % PROGRESS_EVERY_N_BATCHES == 0:
                print(
                    f"  train epoch {epoch:03d} "
                    f"batch {batch_idx:04d}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"iou={metrics['iou']:.4f} "
                    f"f1={metrics['f1']:.4f}",
                    flush=True,
                )

        train_batches = len(train_loader)

        train_loss = train_loss_sum / train_batches
        train_metrics = {
            key: value / train_batches
            for key, value in train_metrics_sum.items()
        }

        val_default = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            DEFAULT_THRESHOLD,
        )

        best_threshold_epoch, threshold_results = find_best_threshold(
            model,
            val_loader,
            device,
            THRESHOLD_CANDIDATES,
        )

        val_best = threshold_results[best_threshold_epoch]

        if scheduler is not None:
            scheduler.step(val_best["iou"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:03d}/{NUM_EPOCHS} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_default['loss']:.4f} "
            f"val_iou@0.50={val_default['iou']:.4f} "
            f"val_f1@0.50={val_default['f1']:.4f} "
            f"best_thr={best_threshold_epoch:.2f} "
            f"val_iou@best={val_best['iou']:.4f} "
            f"val_f1@best={val_best['f1']:.4f} "
            f"precision@best={val_best['precision']:.4f} "
            f"recall@best={val_best['recall']:.4f}",
            flush=True,
        )

        append_history_csv(
            HISTORY_PATH,
            [
                epoch,
                current_lr,
                train_loss,
                train_metrics["iou"],
                train_metrics["f1"],
                train_metrics["precision"],
                train_metrics["recall"],
                val_default["loss"],
                val_default["iou"],
                val_default["f1"],
                val_default["precision"],
                val_default["recall"],
                best_threshold_epoch,
                val_best["iou"],
                val_best["f1"],
                val_best["precision"],
                val_best["recall"],
            ],
        )

        save_prediction_debug(
            model,
            val_dataset,
            device,
            epoch,
            MODEL_TYPE,
            best_threshold_epoch,
            max_samples=DEBUG_SAMPLES_PER_EPOCH,
        )

        checkpoint = {
            "epoch": epoch,
            "model_type": MODEL_TYPE,
            "experiment_name": EXPERIMENT_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou_default_threshold": val_default["iou"],
            "val_f1_default_threshold": val_default["f1"],
            "best_threshold_epoch": best_threshold_epoch,
            "val_iou_best_threshold": val_best["iou"],
            "val_f1_best_threshold": val_best["f1"],
            "val_precision_best_threshold": val_best["precision"],
            "val_recall_best_threshold": val_best["recall"],
            "train_height": TRAIN_HEIGHT,
            "train_width": TRAIN_WIDTH,
            "input_channels": 4,
            "default_threshold": DEFAULT_THRESHOLD,
            "threshold_candidates": THRESHOLD_CANDIDATES,
            "pos_weight": POS_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "current_lr": current_lr,
        }

        last_path = MODEL_DIR / "bev_vehicle_segmentation_last.pth"
        torch.save(checkpoint, last_path)

        if val_best["iou"] > best_val_iou:
            best_val_iou = val_best["iou"]
            best_threshold_overall = best_threshold_epoch

            best_path = MODEL_DIR / "bev_vehicle_segmentation_best.pth"
            torch.save(checkpoint, best_path)

            print("  saved best:", best_path, flush=True)

    print("", flush=True)
    print("training complete", flush=True)
    print("model type:", MODEL_TYPE, flush=True)
    print("best val IoU:", best_val_iou, flush=True)
    print("best threshold:", best_threshold_overall, flush=True)
    print("model saved to:", MODEL_DIR, flush=True)
    print("predictions saved to:", PRED_DIR, flush=True)
    print("history saved to:", HISTORY_PATH, flush=True)


if __name__ == "__main__":
    main()
