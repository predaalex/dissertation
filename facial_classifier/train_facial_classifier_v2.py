import builtins
import time
from datetime import datetime

import numpy as np
import torch
from pytorch_model_summary import summary

import wandb
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{timestamp}] ", *args, **kwargs)


def get_timestamp():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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


def get_dataloaders(base):
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
        batch_size=base['BATCH_SIZE'],
        shuffle=True,
        num_workers=base['NUM_WORKERS'],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=base['BATCH_SIZE'],
        shuffle=False,
        num_workers=base['NUM_WORKERS'],
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=base['BATCH_SIZE'],
        shuffle=False,
        num_workers=base['NUM_WORKERS'],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def get_model(device, config):
    """
    mode:
        - "head" -> train only classifier
        - "full" -> train entire network
    """
    mode = config["TRAINING_MODE"]
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, config["NUM_CLASSES"])

    if mode == "head":
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only classifier
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif mode == "full":
        # Train everything
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError("mode must be 'head' or 'full'")

    return model.to(device)


def compute_normalized_cm(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = cm.astype(np.float32)

    # normalize per row (true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    return cm, cm_norm


def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.tight_layout()
    return fig


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, run, global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        if run is not None and (global_step % 50 == 0):
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/grad_norm": float(grad_norm),
                },
                step=global_step,
            )

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        global_step += 1

    return running_loss / total, correct / total, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_details=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    if return_details:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=list(range(len(CLASS_NAMES))),
            average=None,
            zero_division=0,
        )

        return (
            running_loss / total,
            correct / total,
            macro_f1,
            all_preds,
            all_labels,
            precision,
            recall,
            f1,
            support,
        )

    return (
        running_loss / total,
        correct / total,
        macro_f1,
    )


def build_per_class_table(precision, recall, f1, support):
    return wandb.Table(
        columns=["class", "precision", "recall", "f1", "support"],
        data=[
            [
                CLASS_NAMES[i],
                float(precision[i]),
                float(recall[i]),
                float(f1[i]),
                int(support[i]),
            ]
            for i in range(len(CLASS_NAMES))
        ],
    )


def main(config, train_loader, val_loader, test_loader, device):
    run = wandb.init(
        project="POWERFUL_DISSERTATION",
        name=config["RUN_NAME"],
        config=config
    )

    model = get_model(device, config)
    print(summary(model, torch.rand(size=(config["BATCH_SIZE"], 3, 224, 224)).to(device), show_input=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val_f1 = 0.0
    best_epoch = 0
    global_step = 0

    for epoch in range(config["EPOCHS"]):
        start_time = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, run, global_step
        )

        (
            val_loss,
            val_acc,
            val_f1,
            val_preds,
            val_labels,
            val_precision,
            val_recall,
            val_f1_per_class,
            val_support,
        ) = evaluate(model, val_loader, criterion, device, return_details=True)

        scheduler.step(val_f1)

        end_time = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{config['EPOCHS']}] | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "lr": lr_now,
                "time/epoch_sec": end_time - start_time,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/f1": val_f1,
                "val/per_class_metrics": build_per_class_table(
                    val_precision, val_recall, val_f1_per_class, val_support
                ),
            }

            for i, class_name in enumerate(CLASS_NAMES):
                log_dict[f"val_per_class/precision_{class_name}"] = float(val_precision[i])
                log_dict[f"val_per_class/recall_{class_name}"] = float(val_recall[i])
                log_dict[f"val_per_class/f1_{class_name}"] = float(val_f1_per_class[i])

            wandb.log(log_dict, step=global_step)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "global_step": global_step,
                    "config": config,
                },
                MODEL_PATH,
            )
            print("Saved best model.")

            cm = confusion_matrix(
                val_labels,
                val_preds,
                labels=list(range(len(CLASS_NAMES))),
            )
            fig_norm = plot_confusion_matrix(cm, CLASS_NAMES, normalize=True)

            if run is not None:
                wandb.log(
                    {
                        "best/epoch": best_epoch,
                        "best/val_f1": best_val_f1,
                        "val/confusion_matrix": wandb.Image(fig_norm),
                    },
                    step=global_step,
                )

            plt.close(fig_norm)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    (
        test_loss,
        test_acc,
        test_f1,
        test_preds,
        test_labels,
        test_precision,
        test_recall,
        test_f1_per_class,
        test_support,
    ) = evaluate(model, test_loader, criterion, device, return_details=True)

    cm = confusion_matrix(
        test_labels,
        test_preds,
        labels=list(range(len(CLASS_NAMES))),
    )
    fig_norm = plot_confusion_matrix(cm, CLASS_NAMES, normalize=True)

    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}")

    if run is not None:
        log_dict = {
            "test/loss": test_loss,
            "test/acc": test_acc,
            "test/f1": test_f1,
            "test/per_class_metrics": build_per_class_table(
                test_precision, test_recall, test_f1_per_class, test_support
            ),
            "test/confusion_matrix": wandb.Image(fig_norm),
            "final/best_epoch": best_epoch,
            "final/best_val_f1": best_val_f1,
        }

        for i, class_name in enumerate(CLASS_NAMES):
            log_dict[f"test_per_class/precision_{class_name}"] = float(test_precision[i])
            log_dict[f"test_per_class/recall_{class_name}"] = float(test_recall[i])
            log_dict[f"test_per_class/f1_{class_name}"] = float(test_f1_per_class[i])

        wandb.log(log_dict, step=global_step)

        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_f1"] = best_val_f1
        run.summary["test_loss"] = test_loss
        run.summary["test_acc"] = test_acc
        run.summary["test_f1"] = test_f1

        for i, class_name in enumerate(CLASS_NAMES):
            run.summary[f"test_precision_{class_name}"] = float(test_precision[i])
            run.summary[f"test_recall_{class_name}"] = float(test_recall[i])
            run.summary[f"test_f1_{class_name}"] = float(test_f1_per_class[i])

        run.finish()

    plt.close(fig_norm)
    return {"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1}


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    mp.freeze_support()
    MODEL_PATH = "models/best_mobilenetv2_fer2013.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    with open("api_key.txt", "r") as f:
        api_key = f.readline().strip()
        wandb.login(key=api_key)

    sweep = [
        {"RUN_NAME": "FULL_TRAIN", "TRAINING_MODE": "full", "EPOCHS": 4, "LR": 1e-4, "WEIGHT_DECAY": 1e-4},
    ]

    base = {
        "NUM_CLASSES": 7,
        "BATCH_SIZE": 64,
        "NUM_WORKERS": 0,
    }

    train_loader, val_loader, test_loader = get_dataloaders(base)

    for cfg in sweep:
        config = {**base, **cfg}
        best_results = main(config, train_loader, val_loader, test_loader, device)
        print(
            f"DONE. Best results:\n"
            f"loss={best_results['test_loss']:.4f},\n"
            f"acc={best_results['test_acc']:.4f},\n"
            f"f1={best_results['test_f1']:.4f}\n"
        )
