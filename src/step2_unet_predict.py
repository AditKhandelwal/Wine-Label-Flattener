# src/step2_unet_predict.py
"""
Run the trained U-Net on a photo and save the predicted label mask.
Usage:
    python src/step2_unet_predict.py --img outputs/step1_resized.jpg
"""

import argparse, os
from pathlib import Path
import cv2, torch
from torchvision.transforms.functional import to_tensor, resize
from train.unet import UNet    # ← imports the model you just trained
import numpy as np
import matplotlib.pyplot as plt

def predict_mask(img_path, model_path="checkpoints/unet.pt",
                 out_path="outputs/step2_mask.png", size=256):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. load image ────────────────────────────────────────────────
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 2. prepare tensor (HWC → CHW, 0-1) ───────────────────────────
    t = resize(to_tensor(img_rgb), (size, size)).unsqueeze(0).to(device)

    # ── 3. load model & infer ────────────────────────────────────────
    net = UNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    with torch.no_grad():
        pred = net(t)[0, 0].cpu().numpy()          # (256×256), 0-1

    # ── 4. resize mask back to original image size ───────────────────
    mask = cv2.resize(pred, (img_rgb.shape[1], img_rgb.shape[0]))
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    # ── 5. save & visualise ──────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, mask_bin)
    print(f"[✓] Saved predicted mask → {out_path}")

    plt.subplot(1,2,1); plt.imshow(img_rgb); plt.title("Photo"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(mask_bin, cmap="gray"); plt.title("Predicted Mask"); plt.axis("off")
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="outputs/step1_resized.jpg")
    args = ap.parse_args()
    predict_mask(args.img)
