from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


models_dir = Path("/Users/yasinsensoy/kitti_object/processed_bev/models")

plain_path = models_dir / "bev_vehicle_plain_cnn_352x256_e20_pw8p0" / "history.csv"
unet_path = models_dir / "bev_vehicle_small_unet_448x320_e20_pw8p0" / "history.csv"

plain = pd.read_csv(plain_path)
unet = pd.read_csv(unet_path)

plt.figure()
plt.plot(plain["epoch"], plain["val_iou_best_threshold"], label="Plain CNN")
plt.plot(unet["epoch"], unet["val_iou_best_threshold"], label="Small U-Net")
plt.xlabel("Epoch")
plt.ylabel("Validation IoU")
plt.title("Validation IoU Comparison")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(plain["epoch"], plain["val_f1_best_threshold"], label="Plain CNN")
plt.plot(unet["epoch"], unet["val_f1_best_threshold"], label="Small U-Net")
plt.xlabel("Epoch")
plt.ylabel("Validation F1")
plt.title("Validation F1 Comparison")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(plain["epoch"], plain["train_loss"], label="Plain CNN Train Loss")
plt.plot(plain["epoch"], plain["val_loss"], label="Plain CNN Val Loss")
plt.plot(unet["epoch"], unet["train_loss"], label="U-Net Train Loss")
plt.plot(unet["epoch"], unet["val_loss"], label="U-Net Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

print("\nPlain CNN best row:")
print(plain.loc[plain["val_iou_best_threshold"].idxmax()])

print("\nSmall U-Net best row:")
print(unet.loc[unet["val_iou_best_threshold"].idxmax()])
