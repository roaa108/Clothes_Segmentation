
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2

class_names = ["background", "upper", "lower", "dress"]
num_classes = len(class_names)

# one colour per class
palette = np.array([
    [0, 0, 0],
    [230, 85, 13],
    [49, 130, 189],
    [117, 107, 177],
], dtype=np.uint8)

# ATR has 18 labels 4 upperclothes, 5 skirt, 6 pants, 7 dress.
# everything else (skin, hair, face, hat, belt, shoes, scarf, bag) becomes background.
label_map = np.zeros(256, dtype=np.uint8)
label_map[4] = 1
label_map[5] = 2
label_map[6] = 2
label_map[7] = 3

# image net normalization values, used in the pretrained encoder and in the augmentations.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def train_transform(image_size):
    return A.Compose([
        A.LongestMaxSize(image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, value=0, mask_value=0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])


def eval_transform(image_size):
    return A.Compose([
        A.LongestMaxSize(image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, value=0, mask_value=0),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])


class ATRClothes(Dataset):

    def __init__(self, root, split, transform):
        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.transform = transform

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"{self.image_dir} not found, run download_data.py first")

        self.names = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image = np.asarray(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        mask = label_map[np.asarray(Image.open(os.path.join(self.mask_dir, name)))]

        sample = self.transform(image=image, mask=mask)

        # class indices, so long and not float
        return sample["image"], sample["mask"].long(), name


def atr_dataloader(data_root="./data/atr", image_size=256, batch_size=16,
                   validation_ratio=0.1, seed=42, num_workers=2):
   
    train_source = ATRClothes(data_root, "train", train_transform(image_size))
    val_source = ATRClothes(data_root, "train", eval_transform(image_size))
    test_data = ATRClothes(data_root, "test", eval_transform(image_size))

    indices = np.random.default_rng(seed).permutation(len(train_source))
    validation_count = int(len(indices) * validation_ratio)

    val_data = Subset(val_source, indices[:validation_count])
    train_data = Subset(train_source, indices[validation_count:])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = atr_dataloader()
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    images, masks, names = next(iter(train_loader))
    print(images.shape, masks.shape, masks.dtype)

    # share of every class, to confirm the label map matches the mask files
    for index, name in enumerate(class_names):
        print(name, round((masks == index).float().mean().item(), 4))
