# train/train.py
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import LabelSet
from unet import UNet


def dice_loss(pred, target, eps: float = 1e-6):
    """
    Soft Dice loss — good for imbalanced foreground/background.
    """
    num = (pred * target).sum(dim=(2, 3)) * 2.0
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


def train(img_dir: str, mask_dir: str, epochs: int = 25,
          batch: int = 8, lr: float = 1e-3,
          ckpt_path: str = "checkpoints/unet.pt"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    ds = LabelSet(img_dir, mask_dir)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4)

    net = UNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        net.train()
        running = 0.0
        for imgs, masks in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = net(imgs)
            loss = dice_loss(preds, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)

        print(f"Epoch {ep:02}/{epochs} - loss: {running / len(ds):.4f}")

    # save
    Path(ckpt_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(net.state_dict(), ckpt_path)
    print(f"✓ Model saved to {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    train(args.img_dir, args.mask_dir, args.epochs, args.batch)
