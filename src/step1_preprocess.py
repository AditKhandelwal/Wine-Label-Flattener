import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_image(img_path: str, scale: float = 0.5):
    img_path = Path(img_path).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"{img_path} does not exist")

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cv2.imread could not decode {img_path}")

    # resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (int(w * scale), int(h * scale)))

    # save
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "step1_resized.jpg"
    cv2.imwrite(str(out_file), cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved resized image to {out_file}")

    # preview
    plt.imshow(img_small); plt.axis("off"); plt.title("Step 1"); plt.show()

if __name__ == "__main__":
    # run FROM repo root:  python src/step1_preprocess.py
    preprocess_image("data/image.png", scale=0.5)
