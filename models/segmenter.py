import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from dataset import class_names, num_classes
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

def build_metrics(device):
    return MetricCollection({
        "iou": MulticlassJaccardIndex(num_classes=num_classes, average=None),
        "dice": MulticlassF1Score(num_classes=num_classes, average=None),
        "pixel_accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "confusion": MulticlassConfusionMatrix(num_classes=num_classes),
    }).to(device)


#build dictionary for history and json.
def flatten_metrics(computed):      
    iou = computed["iou"]
    dice = computed["dice"]

    metrics = {}
    for index, name in enumerate(class_names):
        metrics["iou_" + name] = iou[index].item()
        metrics["dice_" + name] = dice[index].item()

    # metrics calculated for clothes only
    metrics["garment_iou"] = iou[1:].mean().item()
    metrics["garment_dice"] = dice[1:].mean().item()
    metrics["pixel_accuracy"] = computed["pixel_accuracy"].item()

    return metrics



class ClothesSegmenter(nn.Module):
    def __init__(self, encoder_weights="imagenet"):
        super().__init__()

        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True
        )

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, images, masks):
        logits = self.forward(images)

        ce_loss = self.cross_entropy(logits, masks)
        dice_loss = self.dice(logits, masks)
        loss = ce_loss + dice_loss

        metrics = {
            "cross_entropy_loss": ce_loss.item(),
            "dice_loss": dice_loss.item(),
            "total_loss": loss.item(),
        }

        return loss, metrics, logits

    @torch.no_grad()
    def predict(self, images):
        logits = self.forward(images)
        return logits.argmax(dim=1)
