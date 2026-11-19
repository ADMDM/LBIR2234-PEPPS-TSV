import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_manifest, rgb_read, mask_read


def letterbox(image, height, width, mask=None):
    """Preserve aspect ratio; padding ignored in loss, removed at inference."""
    if min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError("height et width doivent être des multiples de 32, >= 32")
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (height - nh) // 2, (width - nw) // 2
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    output = np.full((height, width, 3), (124, 116, 104), dtype=np.uint8)
    output[top:top+nh, left:left+nw] = resized
    target = None
    if mask is not None:
        if mask.shape != (h, w):
            raise ValueError("Dimensions image/masque différentes")
        target = np.full((height, width), 255, np.uint8)
        target[top:top+nh, left:left+nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    return output, target, (top, left, nh, nw)


def normalize(image):
    x = image.astype(np.float32) / 255
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def augmentation(seed):
    # No horizontal flips: TSV left/right semantics must not silently swap.
    return A.Compose([
        A.Affine(scale=(.97, 1.03), translate_percent=(-.015, .015), rotate=(-2, 2),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=cv2.BORDER_CONSTANT, fill=(124, 116, 104), fill_mask=255, p=.25),
        A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15, p=.5),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=8, p=.2),
        A.GaussianBlur(blur_limit=(3, 3), p=.15),
    ], seed=seed)


class RailDataset(Dataset):
    def __init__(self, manifest, classes=2, height=384, width=640, train=False, seed=42):
        self.rows = read_manifest(manifest)
        self.height, self.width, self.classes = height, width, classes
        self.transform = augmentation(seed) if train else None
        # Fail before training instead of discovering corrupt IDs mid-epoch.
        self.valid_pixels = 0
        for row in self.rows:
            mask = mask_read(row["mask"], [*range(classes), 255])
            from PIL import Image
            with Image.open(row["image"]) as im:
                if im.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Dimensions différentes : {row['image']}")
            self.valid_pixels += int((mask != 255).sum())
        if self.valid_pixels == 0:
            raise ValueError("Dataset entièrement ignore=255")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image, mask, _ = letterbox(rgb_read(row["image"]), self.height, self.width,
                                   mask_read(row["mask"], [*range(self.classes), 255]))
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        return normalize(image), torch.from_numpy(mask.astype(np.int64))
