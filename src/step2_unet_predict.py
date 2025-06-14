import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def simulate_unet_prediction(img_path, save_path="outputs/step2_mask.png"):
    img_path = Path(img_path)
    save_path = Path(save_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path.resolve()}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image from {img_path}")

    h, w = img.shape[:2]

    # Create dummy mask
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.25), int(h * 0.3))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Ensure output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_path), mask)
    print(f"[✓] Saved simulated mask to: {save_path.resolve()}")

    # Show the mask
    plt.imshow(mask, cmap="gray")
    plt.title("Step 2: Simulated U-Net Label Mask")
    plt.axis("off")
    plt.show()

    return mask

if __name__ == "__main__":
    simulate_unet_prediction("outputs/step1_resized.jpg")
