import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def get_label_points(mask_path, save_vis="outputs/step3_points.png"):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    H, W = mask.shape
    mid_y = H * 0.5                     

    # Find candidate contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    label_cnt = None
    best_area = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if y < mid_y:                 # skip blobs in upper half
            continue
        if w <= h:                    # skip tall / square blobs
            continue
        area = cv2.contourArea(c)
        if area > best_area:
            best_area, label_cnt = area, c

    if label_cnt is None:
        raise RuntimeError("No contour looks like a label")

    # 6 key points on that bounding box 
    x, y, w, h = cv2.boundingRect(label_cnt)
    TL, TR = (x, y),          (x+w, y)
    ML, MR = (x, y+h//2),     (x+w, y+h//2)
    BL, BR = (x, y+h),        (x+w, y+h)
    pts = np.array([TL, TR, ML, MR, BL, BR], dtype=np.int32)

    np.save("outputs/step3_points.npy", pts)

    # Visualise 
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (px, py) in pts:
        cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
    Path(save_vis).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(save_vis, vis)
    print(f"[✓] saved → {save_vis}")

    plt.imshow(vis); plt.title("Step 3: 6 Label Edge Points"); plt.axis("off")
    plt.show()
    return pts

if __name__ == "__main__":
    get_label_points("outputs/step2_mask.png")
