import json
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import atr_dataloader, class_names
from inference import remove_padding, save_results
from models.segmenter import ClothesSegmenter, build_metrics, flatten_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "checkpoints/unet/best.pt"
data_root = "./data/atr"

examples_dir = "results/examples"
metrics_dir = "results/metrics"
os.makedirs(examples_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    metrics = build_metrics(device)

    total_loss = 0
    num_batches = 0
    saved = 0
    max_examples = 8

    for images, masks, names in loader:
        images = images.to(device)
        masks = masks.to(device)

        loss, _, logits = model.compute_loss(images, masks)

        total_loss += loss.item()
        num_batches += 1
        metrics.update(logits, masks)

        if saved < max_examples:
            predictions = logits.argmax(dim=1).cpu().numpy().astype("uint8")

            for prediction, name in zip(predictions, names):
                if saved >= max_examples:
                    break

                original = Image.open(
                    os.path.join(data_root, "test", "images", name)
                )

                # التوقع لسه مربع 256 بالـ padding، لازم يتشال قبل ما يترسم
                # على الصورة الأصلية وإلا الماسك هيطلع مزحزح
                prediction = remove_padding(prediction, original.size)

                save_results(original, prediction, examples_dir, os.path.splitext(name)[0])
                saved += 1

    computed = metrics.compute()
    results = flatten_metrics(computed)
    results["total_loss"] = total_loss / num_batches

    confusion = computed["confusion"].cpu().tolist()
    return results, confusion


def save_training_plot(history):
    epochs = range(1, len(history["train_total_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_total_loss"], label="train")
    plt.plot(epochs, history["validation_total_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_garment_iou"], label="train")
    plt.plot(epochs, history["validation_garment_iou"], label="validation")
    plt.title("Garment IoU")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "training_curves.png"))
    plt.close()


def main():
    _, _, test_loader = atr_dataloader(
        data_root=data_root,
        image_size=256,
        batch_size=16,
    )

    model = ClothesSegmenter(encoder_weights=None).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results, confusion = evaluate(model, test_loader)
    save_training_plot(checkpoint["history"])

    with open(os.path.join(metrics_dir, "results_table.md"), "w") as file:
        file.write("| Class | IoU | Dice |\n|---|---:|---:|\n")
        for name in class_names:
            file.write(
                f"| {name} | {results['iou_' + name]:.4f} | "
                f"{results['dice_' + name]:.4f} |\n"
            )

        file.write(
            f"| **garment mean** | **{results['garment_iou']:.4f}** | "
            f"**{results['garment_dice']:.4f}** |\n\n"
            f"Pixel accuracy: {results['pixel_accuracy']:.4f}\n"
        )

    final_results = {
        "epoch": checkpoint["epoch"],
        "test_metrics": results,
        # rows for refrence, columns for prediction
        "confusion_matrix": {"class_order": class_names, "rows_are_true": confusion},
    }

    with open(os.path.join(metrics_dir, "final_results.json"), "w") as file:
        json.dump(final_results, file, indent=2)

    print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
    main()
