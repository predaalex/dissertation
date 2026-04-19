import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import f1_score

NUM_CLASSES = 7
BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_PATH = "models/best2_mobilenetv2_fer2013.pt"


def ensure_pil_rgb(img):
    # HF dataset may return list / numpy array / PIL image
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        img = img.astype(np.uint8)

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        elif img.ndim == 3:
            img = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    if not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type: {type(img)}")

    return img.convert("RGB")


class FERDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample["image"]
        label = int(sample["emotion"])

        image = ensure_pil_rgb(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def get_dataloaders():
    dataset = load_dataset("abhilash88/fer2013-enhanced", cache_dir="/datasets/fer2013-enhanced")

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    train_transform, eval_transform = get_transforms()

    train_ds = FERDataset(train_data, transform=train_transform)
    val_ds = FERDataset(val_data, transform=eval_transform)
    test_ds = FERDataset(test_data, transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def get_model(device):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    # Unfreeze only new head
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return (
        running_loss / total,
        correct / total,
        f1_score(all_labels, all_preds, average="macro"),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders()
    model = get_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()