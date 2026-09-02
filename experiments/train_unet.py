import torch

from dataset import atr_dataloader
from models.segmenter import ClothesSegmenter
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_root = "./data/atr"
image_size = 256
batch_size = 16
epochs = 15

LR = 1e-4

train_loader, validation_loader, test_loader = atr_dataloader(
    data_root=data_root,
    image_size=image_size,
    batch_size=batch_size,
    validation_ratio=0.1
)

model = ClothesSegmenter().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    device = device
)

history = trainer.train(
    train_loader=train_loader,
    validation_loader=validation_loader,
    epochs=epochs,
    checkpoint_dir="checkpoints/unet",
)
