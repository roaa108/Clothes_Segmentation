import json
import os
import torch

from models.segmenter import build_metrics, flatten_metrics


class Trainer:

    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.metrics = build_metrics(device)

    def run_epoch(self, loader, training):
        self.model.train(training)

        total_losses = {}
        number_of_batches = 0

        self.metrics.reset()

        for images, masks, _ in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.set_grad_enabled(training):
                loss, losses, logits = self.model.compute_loss(images, masks)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for name, value in losses.items():
                if name not in total_losses:
                    total_losses[name] = 0.0

                total_losses[name] += value

            number_of_batches += 1

            self.metrics.update(logits.detach(), masks)

        metrics = {name: value / number_of_batches for name, value in total_losses.items()}
        metrics.update(flatten_metrics(self.metrics.compute()))

        return metrics

    def train_one_epoch(self, train_loader):
        return self.run_epoch(train_loader, training=True)

    def validate(self, validation_loader):
        return self.run_epoch(validation_loader, training=False)

    def train(self, train_loader, validation_loader, epochs, checkpoint_dir):

        os.makedirs(checkpoint_dir, exist_ok=True)
        history = {}
        best_garment_iou = 0.0

        for epoch in range(1, epochs + 1):

            train_metrics = self.train_one_epoch(train_loader)

            validation_metrics = self.validate(validation_loader)

            for name, value in train_metrics.items():
                history_name = "train_" + name

                if history_name not in history:
                    history[history_name] = []

                history[history_name].append(value)

            for name, value in validation_metrics.items():
                history_name = "validation_" + name

                if history_name not in history:
                    history[history_name] = []

                history[history_name].append(value)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train loss: {train_metrics['total_loss']:.4f} | "
                f"Validation loss: {validation_metrics['total_loss']:.4f} | "
                f"Garment IoU: {validation_metrics['garment_iou']:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": history,
            }

            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "latest.pt"),
            )

            if validation_metrics["garment_iou"] > best_garment_iou:
                best_garment_iou = validation_metrics["garment_iou"]

                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, "best.pt"),
                )

            with open(os.path.join(checkpoint_dir, "history.json"), "w") as file:
                json.dump(history, file, indent=2)

        return history
