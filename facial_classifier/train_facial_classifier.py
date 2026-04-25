import builtins
import copy
import argparse
from collections import Counter
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp
import wandb
from datasets import load_dataset
from PIL import Image
from pytorch_model_summary import summary
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

PROJECT_NAME = "POWERFUL_DISSERTATION"
DEFAULT_DATASET_FRACTION = 0.1
DATASET_CACHE_DIR = "datasets/fer2013-enhanced"
MODELS_DIR = Path("models")


def print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{timestamp}] ", *args, **kwargs)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_pil_rgb(img):
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
    def __init__(
        self,
        hf_dataset,
        transform=None,
        fraction=DEFAULT_DATASET_FRACTION,
        min_quality_score=None,
    ):
        self.transform = transform
        if min_quality_score is not None:
            hf_dataset = hf_dataset.filter(
                lambda sample: float(sample.get("quality_score", 0.0)) >= float(min_quality_score)
            )
        if fraction is None or fraction >= 1.0:
            self.ds = hf_dataset
        else:
            n = max(1, int(len(hf_dataset) * fraction))
            self.ds = hf_dataset.shuffle(seed=42).select(range(n))

        self.labels = [int(label) for label in self.ds["emotion"]]
        if "sample_weight" in self.ds.column_names:
            self.sample_weights = [float(weight) for weight in self.ds["sample_weight"]]
        else:
            counts = Counter(self.labels)
            self.sample_weights = [1.0 / counts[label] for label in self.labels]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = ensure_pil_rgb(sample["image"])
        label = int(sample["emotion"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((232, 232)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((232, 232)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, eval_transform


def get_dataloaders(config):
    dataset = load_dataset("abhilash88/fer2013-enhanced", cache_dir=DATASET_CACHE_DIR)

    train_transform, eval_transform = get_transforms()
    fraction = float(config.get("dataset_fraction", DEFAULT_DATASET_FRACTION))
    min_quality_score = config.get("min_quality_score")

    train_ds = FERDataset(
        dataset["train"],
        transform=train_transform,
        fraction=fraction,
        min_quality_score=min_quality_score,
    )
    val_ds = FERDataset(
        dataset["validation"],
        transform=eval_transform,
        fraction=fraction,
        min_quality_score=min_quality_score,
    )
    test_ds = FERDataset(
        dataset["test"],
        transform=eval_transform,
        fraction=fraction,
        min_quality_score=min_quality_score,
    )

    common_loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }

    imbalance_strategy = str(config.get("imbalance_strategy", "class_weighted_loss"))
    train_loader_kwargs = dict(common_loader_kwargs)

    if imbalance_strategy == "weighted_sampler":
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(train_ds.sample_weights),
            num_samples=len(train_ds.sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, **train_loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs)

    return train_loader, val_loader, test_loader


def create_model(config, device):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    in_features = model.classifier[1].in_features
    dropout = float(config.get("dropout", 0.2))
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, int(config["num_classes"])),
    )
    return model.to(device)


def freeze_backbone_train_head(model):
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model, unfreeze_from_block):
    freeze_backbone_train_head(model)
    total_blocks = len(model.features)
    block_start = max(0, min(int(unfreeze_from_block), total_blocks - 1))

    for block_idx in range(block_start, total_blocks):
        for param in model.features[block_idx].parameters():
            param.requires_grad = True

    return block_start


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_optimizer(model, lr, weight_decay):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )


def build_criterion(config):
    class_weights = config.get("class_weights")
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )


def compute_class_weights(labels, num_classes):
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = []

    for class_idx in range(num_classes):
        count = counts.get(class_idx, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    return weights, counts


def summarize_class_distribution(labels):
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    summary = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        count = counts.get(class_idx, 0)
        pct = (100.0 * count / total) if total else 0.0
        summary[class_name] = {"count": count, "pct": pct}

    return summary


def use_amp(device):
    return device.type == "cuda"


def get_amp_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


def build_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=use_amp(device))


def compute_normalized_cm(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(np.float32)
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


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    global_step,
    config,
    phase_name,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_clip = float(config.get("grad_clip", 1.0))

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_amp_context(device):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if global_step % 50 == 0:
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/grad_norm": float(grad_norm),
                    "phase/name": phase_name,
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

        with get_amp_context(device):
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

    return running_loss / total, correct / total, macro_f1


def build_checkpoint_name(config, run_name, run_id, best_val_f1):
    timestamp = get_timestamp()
    strategy = config["strategy"]
    lr = format(float(config["lr"]), ".0e")
    batch_size = int(config["batch_size"])
    dataset_fraction = float(config.get("dataset_fraction", DEFAULT_DATASET_FRACTION))
    score = f"{best_val_f1:.4f}".replace(".", "p")
    safe_run_name = str(run_name).replace(" ", "_")
    filename = (
        f"{timestamp}_{strategy}_{safe_run_name}_{run_id}_"
        f"bs{batch_size}_lr{lr}_frac{dataset_fraction:.2f}_f1{score}.pt"
    )
    return MODELS_DIR / filename


def latest_checkpoint_path(strategy):
    return MODELS_DIR / f"latest_{strategy}.pt"


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    config,
    run,
    best_state,
    phase_name,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = build_checkpoint_name(
        config,
        run.name,
        run.id,
        best_state["best_val_f1"],
    )
    checkpoint = {
        "epoch": best_state["epoch"],
        "phase_name": phase_name,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_state["best_val_f1"],
        "best_val_loss": best_state["best_val_loss"],
        "best_val_acc": best_state["best_val_acc"],
        "global_step": best_state["global_step"],
        "config": dict(config),
        "checkpoint_path": str(checkpoint_path),
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
    }
    torch.save(checkpoint, checkpoint_path)
    shutil.copyfile(checkpoint_path, latest_checkpoint_path(config["strategy"]))
    return checkpoint_path


def log_epoch_metrics(
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    val_f1,
    val_precision,
    val_recall,
    val_f1_per_class,
    val_support,
    optimizer,
    phase_name,
    phase_epoch,
    total_phase_epochs,
    global_step,
    trainable_params,
    unfreeze_from_block,
):
    wandb.log(
        {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "val/per_class_metrics": build_per_class_table(
                val_precision, val_recall, val_f1_per_class, val_support
            ),
            "train/lr": optimizer.param_groups[0]["lr"],
            "phase/name": phase_name,
            "phase/epoch": phase_epoch,
            "phase/epochs_total": total_phase_epochs,
            "model/trainable_params": trainable_params,
            "model/unfreeze_from_block": (
                -1 if unfreeze_from_block is None else int(unfreeze_from_block)
            ),
        },
        step=global_step,
    )


def train_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    scaler,
    config,
    run,
    phase_name,
    num_epochs,
    global_step,
    best_state,
    checkpoint_path,
    unfreeze_from_block=None,
):
    trainable_params = count_trainable_parameters(model)

    for epoch_idx in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            global_step=global_step,
            config=config,
            phase_name=phase_name,
        )
        (
            val_loss,
            val_acc,
            val_f1,
            _val_preds,
            _val_labels,
            val_precision,
            val_recall,
            val_f1_per_class,
            val_support,
        ) = evaluate(model, val_loader, criterion, device, return_details=True)

        scheduler.step(val_f1)
        elapsed = time.time() - start_time
        epoch_number = epoch_idx + 1

        print(
            f"{phase_name} [{epoch_number}/{num_epochs}] | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} trainable={trainable_params} | "
            f"time={elapsed:.1f}s"
        )

        log_epoch_metrics(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1_per_class=val_f1_per_class,
            val_support=val_support,
            optimizer=optimizer,
            phase_name=phase_name,
            phase_epoch=epoch_number,
            total_phase_epochs=num_epochs,
            global_step=global_step,
            trainable_params=trainable_params,
            unfreeze_from_block=unfreeze_from_block,
        )

        if val_f1 > best_state["best_val_f1"]:
            best_state["best_val_f1"] = val_f1
            best_state["best_val_loss"] = val_loss
            best_state["best_val_acc"] = val_acc
            best_state["epoch"] = epoch_number
            best_state["global_step"] = global_step
            best_state["phase_name"] = phase_name
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                run=run,
                best_state=best_state,
                phase_name=phase_name,
            )
            print(f"Saved best model to {checkpoint_path}.")

    return best_state, global_step, checkpoint_path


def fit_baseline(model, train_loader, val_loader, criterion, config, device, run):
    freeze_backbone_train_head(model)

    optimizer = build_optimizer(model, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = build_scheduler(optimizer)
    scaler = build_scaler(device)
    best_state = {
        "best_val_f1": float("-inf"),
        "best_val_loss": None,
        "best_val_acc": None,
        "epoch": 0,
        "global_step": 0,
        "phase_name": "head",
    }

    return train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        config=config,
        run=run,
        phase_name="head",
        num_epochs=int(config["epochs"]),
        global_step=0,
        best_state=best_state,
        checkpoint_path=None,
        unfreeze_from_block=None,
    )


def fit_finetune(model, train_loader, val_loader, criterion, config, device, run):
    freeze_backbone_train_head(model)
    scaler = build_scaler(device)
    best_state = {
        "best_val_f1": float("-inf"),
        "best_val_loss": None,
        "best_val_acc": None,
        "epoch": 0,
        "global_step": 0,
        "phase_name": "head",
    }
    global_step = 0
    checkpoint_path = None

    freeze_epochs = int(config["freeze_epochs"])
    if freeze_epochs > 0:
        head_optimizer = build_optimizer(
            model,
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        head_scheduler = build_scheduler(head_optimizer)
        best_state, global_step, checkpoint_path = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=head_optimizer,
            scheduler=head_scheduler,
            device=device,
            scaler=scaler,
            config=config,
            run=run,
            phase_name="head",
            num_epochs=freeze_epochs,
            global_step=global_step,
            best_state=best_state,
            checkpoint_path=checkpoint_path,
            unfreeze_from_block=None,
        )

    block_start = unfreeze_last_blocks(model, int(config["unfreeze_from_block"]))
    finetune_optimizer = build_optimizer(
        model,
        lr=float(config["finetune_lr"]),
        weight_decay=float(config["finetune_weight_decay"]),
    )
    finetune_scheduler = build_scheduler(finetune_optimizer)

    finetune_epochs = int(config["finetune_epochs"])
    if finetune_epochs > 0:
        best_state, global_step, checkpoint_path = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=finetune_optimizer,
            scheduler=finetune_scheduler,
            device=device,
            scaler=scaler,
            config=config,
            run=run,
            phase_name="finetune",
            num_epochs=finetune_epochs,
            global_step=global_step,
            best_state=best_state,
            checkpoint_path=checkpoint_path,
            unfreeze_from_block=block_start,
        )

    return best_state, global_step, checkpoint_path


def evaluate_best_checkpoint(model, test_loader, criterion, device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    cm, _cm_norm = compute_normalized_cm(test_labels, test_preds, num_classes=len(CLASS_NAMES))
    fig_norm = plot_confusion_matrix(cm, CLASS_NAMES, normalize=True)

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1_per_class": test_f1_per_class,
        "test_support": test_support,
        "figure": fig_norm,
        "checkpoint": checkpoint,
    }


def default_run_name(config):
    return (
        f"{config['strategy']}_"
        f"bs{int(config['batch_size'])}_"
        f"lr{float(config['lr']):.0e}_"
        f"frac{float(config['dataset_fraction']):.2f}"
    )


def prepare_run_config(run_config):
    config = dict(run_config)
    config.setdefault("strategy", "baseline")
    config.setdefault("batch_size", 64)
    config.setdefault("num_workers", 0)
    config.setdefault("num_classes", 7)
    config.setdefault("epochs", 10)
    config.setdefault("lr", 1e-4)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("dropout", 0.2)
    config.setdefault("label_smoothing", 0.0)
    config.setdefault("grad_clip", 1.0)
    config.setdefault("dataset_fraction", DEFAULT_DATASET_FRACTION)
    config.setdefault("imbalance_strategy", "class_weighted_loss")
    config.setdefault("min_quality_score", None)
    config.setdefault("freeze_epochs", config["epochs"])
    config.setdefault("finetune_epochs", 5)
    config.setdefault("finetune_lr", 1e-5)
    config.setdefault("finetune_weight_decay", config["weight_decay"])
    config.setdefault("unfreeze_from_block", 14)
    config.setdefault("run_name", default_run_name(config))

    strategy = config["strategy"]
    if strategy not in {"baseline", "finetune"}:
        raise ValueError("strategy must be 'baseline' or 'finetune'")

    imbalance_strategy = config["imbalance_strategy"]
    if imbalance_strategy not in {"none", "class_weighted_loss", "weighted_sampler"}:
        raise ValueError(
            "imbalance_strategy must be 'none', 'class_weighted_loss', or 'weighted_sampler'"
        )

    if strategy == "baseline":
        config["freeze_epochs"] = config["epochs"]
        config["finetune_epochs"] = 0

    return config


def run_training(config_overrides=None):
    config = prepare_run_config(config_overrides or {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project=PROJECT_NAME,
        name=config["run_name"],
        config=config,
        tags=[config["strategy"], "mobilenetv2", "fer2013"],
    )
    config = dict(wandb.config)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    class_weights = None
    if config["imbalance_strategy"] == "class_weighted_loss":
        class_weights, class_counts = compute_class_weights(
            train_loader.dataset.labels,
            num_classes=int(config["num_classes"]),
        )
        config["class_weights"] = class_weights
    else:
        class_counts = Counter(train_loader.dataset.labels)

    class_distribution = summarize_class_distribution(train_loader.dataset.labels)
    model = create_model(config, device)
    example_input = torch.rand(size=(int(config["batch_size"]), 3, 224, 224), device=device)
    print(summary(model, example_input, show_input=True))

    criterion = build_criterion(config)
    criterion = criterion.to(device)

    print(f"Imbalance strategy: {config['imbalance_strategy']}")
    print(f"Class counts: {dict(class_counts)}")
    if class_weights is not None:
        print(f"Class weights: {[round(weight, 4) for weight in class_weights]}")
    if config.get("min_quality_score") is not None:
        print(f"Minimum quality score: {config['min_quality_score']}")

    run.summary["imbalance_strategy"] = config["imbalance_strategy"]
    run.summary["train_class_distribution"] = class_distribution
    if class_weights is not None:
        run.summary["class_weights"] = class_weights

    if config["strategy"] == "baseline":
        best_state, global_step, checkpoint_path = fit_baseline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            device=device,
            run=run,
        )
    else:
        best_state, global_step, checkpoint_path = fit_finetune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            device=device,
            run=run,
        )

    if checkpoint_path is None:
        raise RuntimeError("No checkpoint was created. Check the phase epoch configuration.")

    test_results = evaluate_best_checkpoint(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print(
        f"Test loss={test_results['test_loss']:.4f} "
        f"test_acc={test_results['test_acc']:.4f} "
        f"test_f1={test_results['test_f1']:.4f}"
    )

    wandb.log(
        {
            "test/loss": test_results["test_loss"],
            "test/acc": test_results["test_acc"],
            "test/f1": test_results["test_f1"],
            "test/per_class_metrics": build_per_class_table(
                test_results["test_precision"],
                test_results["test_recall"],
                test_results["test_f1_per_class"],
                test_results["test_support"],
            ),
            "test/confusion_matrix": wandb.Image(test_results["figure"]),
            "summary/best_val_f1": best_state["best_val_f1"],
            "summary/best_phase": best_state["phase_name"],
            "summary/checkpoint_path": str(checkpoint_path),
            "config/imbalance_strategy": config["imbalance_strategy"],
        },
        step=global_step,
    )

    run.summary["checkpoint_path"] = str(checkpoint_path)
    run.summary["best_val_f1"] = best_state["best_val_f1"]
    run.summary["best_phase"] = best_state["phase_name"]
    run.finish()
    plt.close(test_results["figure"])

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_f1": best_state["best_val_f1"],
        "test_loss": test_results["test_loss"],
        "test_acc": test_results["test_acc"],
        "test_f1": test_results["test_f1"],
    }


def sweep_train():
    run_training()


def create_baseline_sweep_config():
    return {
        "method": "bayes",
        "metric": {"name": "val/f1", "goal": "maximize"},
        "parameters": {
            "strategy": {"value": "baseline"},
            "batch_size": {"values": [32, 64]},
            "epochs": {"values": [8, 10, 12]},
            "lr": {"values": [1e-4, 3e-4, 5e-4]},
            "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "dropout": {"values": [0.2, 0.3]},
            "label_smoothing": {"values": [0.0, 0.05, 0.1]},
            "grad_clip": {"values": [0.5, 1.0]},
            "imbalance_strategy": {"values": ["class_weighted_loss", "weighted_sampler"]},
            "dataset_fraction": {"value": DEFAULT_DATASET_FRACTION},
            "min_quality_score": {"values": [None, 0.35, 0.5]},
            "num_workers": {"value": 0},
            "num_classes": {"value": 7},
        },
    }


def create_finetune_sweep_config():
    return {
        "method": "bayes",
        "metric": {"name": "val/f1", "goal": "maximize"},
        "parameters": {
            "strategy": {"value": "finetune"},
            "batch_size": {"values": [32, 64]},
            "freeze_epochs": {"values": [3, 5]},
            "finetune_epochs": {"values": [3, 5, 7]},
            "lr": {"values": [1e-4, 3e-4]},
            "weight_decay": {"values": [1e-5, 1e-4]},
            "finetune_lr": {"values": [1e-5, 3e-5, 5e-5]},
            "finetune_weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "unfreeze_from_block": {"values": [12, 14, 16]},
            "dropout": {"values": [0.2, 0.3]},
            "label_smoothing": {"values": [0.0, 0.05]},
            "grad_clip": {"values": [0.5, 1.0]},
            "imbalance_strategy": {"values": ["class_weighted_loss", "weighted_sampler"]},
            "dataset_fraction": {"value": DEFAULT_DATASET_FRACTION},
            "min_quality_score": {"values": [None, 0.35, 0.5]},
            "num_workers": {"value": 0},
            "num_classes": {"value": 7},
        },
    }


def create_manual_debug_runs():
    return [
        {
            "run_name": "baseline_debug",
            "strategy": "baseline",
            "epochs": 3,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "imbalance_strategy": "class_weighted_loss",
        },
        {
            "run_name": "finetune_debug",
            "strategy": "finetune",
            "freeze_epochs": 2,
            "finetune_epochs": 2,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "finetune_lr": 1e-5,
            "finetune_weight_decay": 1e-4,
            "unfreeze_from_block": 14,
            "imbalance_strategy": "class_weighted_loss",
        },
    ]


def launch_sweep(sweep_type, count=None):
    if sweep_type == "baseline":
        sweep_config = create_baseline_sweep_config()
    elif sweep_type == "finetune":
        sweep_config = create_finetune_sweep_config()
    else:
        raise ValueError("sweep_type must be 'baseline' or 'finetune'")

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    print(f"Created {sweep_type} sweep: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_train, count=count)


def load_wandb_api_key():
    with open("api_key.txt", "r", encoding="utf-8") as file:
        return file.readline().strip()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train FER MobileNetV2 baseline or staged fine-tuning runs."
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "single", "sweep", "agent"],
        default="manual",
        help="manual runs built-in debug configs; single runs one config; sweep creates and runs a W&B sweep; agent executes one sweep-assigned run.",
    )
    parser.add_argument(
        "--strategy",
        choices=["baseline", "finetune"],
        default="baseline",
        help="Training strategy for single runs.",
    )
    parser.add_argument("--run-name", default=None, help="Optional W&B run name override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Baseline total epochs.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--dataset-fraction", type=float, default=None)
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "class_weighted_loss", "weighted_sampler"],
        default=None,
        help="How to address class imbalance in the training split.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=None,
        help="Optional minimum quality_score filter from the enhanced FER2013 dataset.",
    )
    parser.add_argument("--freeze-epochs", type=int, default=None, help="Head-only epochs for fine-tuning.")
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--finetune-weight-decay", type=float, default=None)
    parser.add_argument("--unfreeze-from-block", type=int, default=None)
    parser.add_argument(
        "--sweep-type",
        choices=["baseline", "finetune"],
        default="baseline",
        help="Sweep family to launch when --mode sweep is used.",
    )
    parser.add_argument("--sweep-count", type=int, default=None, help="Optional max number of sweep runs.")
    parser.add_argument(
        "--print-sweep-config",
        action="store_true",
        help="Print the selected sweep config and exit.",
    )
    parser.add_argument(
        "--skip-wandb-login",
        action="store_true",
        help="Skip reading api_key.txt and calling wandb.login().",
    )
    parser.add_argument(
        "--run-manual-finetune",
        type=str2bool,
        default=True,
        help="When --mode manual, include the built-in fine-tune debug run.",
    )
    return parser


def cli_args_to_config(args):
    config = {
        "strategy": args.strategy,
    }

    optional_mappings = {
        "run_name": args.run_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "dataset_fraction": args.dataset_fraction,
        "imbalance_strategy": args.imbalance_strategy,
        "min_quality_score": args.min_quality_score,
        "freeze_epochs": args.freeze_epochs,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "finetune_weight_decay": args.finetune_weight_decay,
        "unfreeze_from_block": args.unfreeze_from_block,
    }

    for key, value in optional_mappings.items():
        if value is not None:
            config[key] = value

    return config


def maybe_login_to_wandb(skip_login):
    if skip_login:
        return
    wandb.login(key=load_wandb_api_key())


def main():
    torch.multiprocessing.freeze_support()
    mp.freeze_support()
    args = build_arg_parser().parse_args()

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    maybe_login_to_wandb(args.skip_wandb_login)

    if args.print_sweep_config:
        sweep_config = (
            create_baseline_sweep_config()
            if args.sweep_type == "baseline"
            else create_finetune_sweep_config()
        )
        builtins.print(sweep_config)
        return

    if args.mode == "manual":
        manual_runs = create_manual_debug_runs()
        if not args.run_manual_finetune:
            manual_runs = [cfg for cfg in manual_runs if cfg["strategy"] == "baseline"]
        for cfg in manual_runs:
            results = run_training(cfg)
            print(
                f"DONE. Best results:\n"
                f"checkpoint={results['checkpoint_path']},\n"
                f"best_val_f1={results['best_val_f1']:.4f},\n"
                f"loss={results['test_loss']:.4f},\n"
                f"acc={results['test_acc']:.4f},\n"
                f"f1={results['test_f1']:.4f}\n"
            )
    elif args.mode == "single":
        results = run_training(cli_args_to_config(args))
        print(
            f"DONE. Best results:\n"
            f"checkpoint={results['checkpoint_path']},\n"
            f"best_val_f1={results['best_val_f1']:.4f},\n"
            f"loss={results['test_loss']:.4f},\n"
            f"acc={results['test_acc']:.4f},\n"
            f"f1={results['test_f1']:.4f}\n"
        )
    elif args.mode == "sweep":
        launch_sweep(sweep_type=args.sweep_type, count=args.sweep_count)
    elif args.mode == "agent":
        sweep_train()
    else:
        raise ValueError("mode must be 'manual', 'single', 'sweep', or 'agent'")


if __name__ == "__main__":
    main()
