import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(img_path, scale=0.5, save_path="../outputs/step1_resized.jpg"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, None, fx=scale, fy=scale)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    # Optional: show
    plt.imshow(resized)
    plt.title("Step 1: Resized Image")
    plt.axis("off")
    plt.show()

    return resized

# Example usage
if __name__ == "__main__":
    preprocess_image("data/image.png", scale=0.5)
    