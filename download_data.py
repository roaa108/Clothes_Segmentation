"""Download ATR from Hugging Face into data/atr/<split>/images and .../masks."""

import os
from datasets import load_dataset
from tqdm import tqdm

output_root = "./data/atr"


def save_split(split_data, split_name, output_root):
    image_dir = os.path.join(output_root, split_name, "images")
    mask_dir = os.path.join(output_root, split_name, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for index, sample in enumerate(tqdm(split_data, desc=split_name)):
        name = f"{index:06d}.png"
        sample["pixel_values"].convert("RGB").save(os.path.join(image_dir, name))
        # the mask keeps the original ATR label ids, the remapping happens in dataset.py
        sample["label"].save(os.path.join(mask_dir, name))


if __name__ == "__main__":
    dataset = load_dataset("ckotait/ATRDataset")
    print("splits:", list(dataset.keys()))

    for split_name, split_data in dataset.items():
        save_split(split_data, split_name, output_root)
