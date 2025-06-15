import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def unwrap_label(img_path: str,
                 pts_path: str,
                 out_path: str,
                 out_w: int = 800,
                 out_h: int = 400):

    #Load Resources
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)

    pts = np.load(pts_path).astype(np.float32)        # TL, TR, ML, MR, BL, BR
    if pts.shape != (6, 2):
        raise ValueError("points file must contain 6×2 array")

    # Choose 4 extreme corners
    TL, TR, _, _, BL, BR = pts
    src = np.array([TL, TR, BR, BL], dtype=np.float32)  # clockwise

    dst = np.array([[0,          0],
                    [out_w - 1,  0],
                    [out_w - 1,  out_h - 1],
                    [0,          out_h - 1]], dtype=np.float32)

    # Perspective transform 
    M = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(bgr, M, (out_w, out_h))

    # Save & Preview
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, flat)
    print(f"[✓] Unwrapped label → {out_path}")

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(bgr,  cv2.COLOR_BGR2RGB))
    plt.title("Original"); plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB))
    plt.title("Flat label"); plt.axis("off")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 4: unwrap label")
    ap.add_argument("--img", default="outputs/step1_resized.jpg")
    ap.add_argument("--pts", default="outputs/step3_points.npy")
    ap.add_argument("--out", default="outputs/step4_flat_label.png")
    ap.add_argument("--w", type=int, default=800, help="output width")
    ap.add_argument("--h", type=int, default=400, help="output height")
    args = ap.parse_args()

    unwrap_label(args.img, args.pts, args.out, args.w, args.h)
