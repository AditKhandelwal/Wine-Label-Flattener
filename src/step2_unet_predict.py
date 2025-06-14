import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def auto_label_mask(img_path, save_path="outputs/step2_mask.png"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]

    # Convert to HSV and work on bottom half only
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_bottom = hsv[h//2:, :]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # Tighter filter for label paper: very bright, low-sat
    lower = np.array([0, 0, 200])
    upper = np.array([180, 40, 255])
    sub_mask = cv2.inRange(hsv_bottom, lower, upper)

    # Find contours in bottom half
    contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No bright region found in bottom half.")

    label_contour = None
    max_score = -1

    # Evaluate best contour by area × aspect ratio score
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000: continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect = w_box / h_box if h_box > 0 else 0

        if 0.8 <= aspect <= 5.0:  # allow more square-like labels too
            score = area * aspect
            if score > max_score:
                max_score = score
                label_contour = cnt

    if label_contour is None:
        raise RuntimeError("No label-like contour passed all filters.")

    # Shift y-coords and draw full mask
    label_contour += np.array([0, h//2])
    cv2.drawContours(full_mask, [label_contour], -1, 255, -1)

    # Save and show
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, full_mask)
    print(f"[✓] Saved final cleaned label mask → {save_path}")

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original"); plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(full_mask, cmap="gray")
    plt.title("Final Mask"); plt.axis("off")

    plt.show()
    return full_mask

if __name__ == "__main__":
    auto_label_mask("data/image.png")
