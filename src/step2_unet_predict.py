import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, resize
import matplotlib.pyplot as plt

# Import Model
from train.unet import UNet      



def predict_mask(image_path: str,
                 model_path: str,
                 out_path: str,
                 input_size: int = 256,
                 thresh: float = 0.5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Prepare Tensor 
    tens = resize(to_tensor(rgb), (input_size, input_size)).unsqueeze(0).to(device)

    # Load Model
    net = UNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Inference
    with torch.no_grad():
        pred = net(tens)[0, 0].cpu().numpy()   

    # Resize back to original resolution
    H, W = rgb.shape[:2]
    pred_full = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_bin = (pred_full > thresh).astype(np.uint8) * 255

    # Cleanup: keep largest blob & smooth morph 
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_bin = (labels == largest).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,  kernel)

    # Save & Preview 
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, mask_bin)
    print(f"[✓] Saved cleaned mask → {out_path}")

    plt.subplot(1, 2, 1)
    plt.imshow(rgb);       plt.title("Input");    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_bin, cmap="gray"); plt.title("Cleaned Mask"); plt.axis("off")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict wine-label mask with trained U-Net")
    ap.add_argument("--img",   default="outputs/step1_resized.jpg", help="input image")
    ap.add_argument("--model", default="checkpoints/unet.pt",       help="model .pt file")
    ap.add_argument("--out",   default="outputs/step2_mask.png",    help="output mask file")
    ap.add_argument("--size",  type=int, default=256, help="network input resolution")
    ap.add_argument("--th",    type=float, default=0.5, help="probability threshold (0-1)")
    args = ap.parse_args()

    predict_mask(args.img, args.model, args.out, args.size, args.th)
