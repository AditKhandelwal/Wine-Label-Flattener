# train/unet.py
import torch
import torch.nn as nn


def _block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.enc1 = _block(3, 16)
        self.enc2 = _block(16, 32)
        self.enc3 = _block(32, 64)
        self.enc4 = _block(64, 128)

        self.pool = nn.MaxPool2d(2)

        # decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = _block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = _block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = _block(32, 16)

        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))

        u3 = self.up3(c4)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        return torch.sigmoid(self.out_conv(d1))
