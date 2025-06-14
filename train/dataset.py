import os
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class LabelSet(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, size: int = 256):
        self.img_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.img_paths = sorted(self.img_dir.glob("*"))
        assert len(self.img_paths) > 0, "No images found!"

        self.to_tensor = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".jpg")   # or ".jpeg" if that’s the exact suffix


        # RGB image
        img = Image.open(img_path).convert("RGB")
        img = self.to_tensor(img)

        # Binary mask (grayscale → 0/1 float tensor)
        mask = Image.open(mask_path).convert("L")
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()   # binarise

        return img, mask
