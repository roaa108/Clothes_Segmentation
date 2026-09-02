import os
import sys
import numpy as np
import torch
from PIL import Image
from dataset import palette, class_names, eval_transform
from models.segmenter import ClothesSegmenter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/unet/best.pt"
image_size = 256


def load_model():
    model = ClothesSegmenter(encoder_weights=None).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def remove_padding(mask, original_size):
    width, height = original_size

    scale = image_size / max(width, height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)

    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2

    return mask[top:top + resized_height, left:left + resized_width]


def save_results(image, mask, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    image = image.convert("RGB")

    # Resize predicted mask back to the original photo size.
    mask = np.asarray(
        Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
    )

    color_mask = Image.fromarray(palette[mask])
    overlay = Image.blend(image, color_mask, alpha=0.45)

    clothes_only = np.asarray(image).copy()
    clothes_only[mask == 0] = 0

    color_mask.save(os.path.join(output_dir, f"{name}_mask.png"))
    overlay.save(os.path.join(output_dir, f"{name}_overlay.png"))
    Image.fromarray(clothes_only).save(
        os.path.join(output_dir, f"{name}_clothing_only.png")
    )


@torch.no_grad()
def predict_mask(model, image_path):
    image = Image.open(image_path).convert("RGB")

    tensor = eval_transform(image_size)(
        image=np.asarray(image)
    )["image"]

    logits = model(tensor.unsqueeze(0).to(device))
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    mask = remove_padding(mask, image.size)

    return image, mask


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [output_dir]")
        return

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/inference"

    model = load_model()
    image, mask = predict_mask(model, image_path)

    name = os.path.splitext(os.path.basename(image_path))[0]
    save_results(image, mask, output_dir, name)

    for index, class_name in enumerate(class_names):
        percentage = (mask == index).mean() * 100
        print(f"{class_name}: {percentage:.1f}%")

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
