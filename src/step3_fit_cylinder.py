import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def get_label_points(mask_path, save_vis="outputs/step3_points.png"):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    # Find external contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    # label_cnt, best_score = None, -1
    # for c in cnts:
    #     x, y, w, h = cv2.boundingRect(c)
    #     if w <= h:                               # skip tall / square blobs
    #         continue
    #     area   = cv2.contourArea(c)
    #     aspect = w / h
    #     # Bias toward large, wide blobs lower in the image
    #     score  = area * aspect * (y + h / 2)
    #     if score > best_score:
    #         best_score, label_cnt = score, c

    label_cnt, best_area = None, -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area > best_area:
            best_area, label_cnt = area, c

    if label_cnt is None:
        raise RuntimeError("No contour looks like a label")

    # Compute 6 key points on the bounding box 
    x, y, w, h = cv2.boundingRect(label_cnt)
    TL, TR = (x, y),          (x + w, y)
    ML, MR = (x, y + h // 2), (x + w, y + h // 2)
    BL, BR = (x, y + h),      (x + w, y + h)
    pts = np.array([TL, TR, ML, MR, BL, BR], dtype=np.int32)

    # Save for Step 4
    np.save("outputs/step3_points.npy", pts)

    # Visualisation
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (px, py) in pts:
        cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)

    Path(save_vis).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_vis, vis)
    print(f"[✓] saved → {save_vis}")

    plt.imshow(vis); plt.title("Step 3: 6 Label Edge Points"); plt.axis("off")
    plt.show()
    return pts


if __name__ == "__main__":
    get_label_points("outputs/step2_mask.png")
