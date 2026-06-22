from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistant_backend.config import Settings  # noqa: E402


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_PERSONAL_ROOT = SCRIPT_DIR / "datasets" / "personal_fer"
DEFAULT_FER_CACHE_ROOT = SCRIPT_DIR / "datasets" / "fer2013-enhanced"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models" / "personal_finetune"


@dataclass
class ImageRecord:
    label: int
    source: str
    weight: float
    path: Path | None = None
    image: Any | None = None


class EmotionImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_record_image(record)
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


def load_record_image(record: ImageRecord) -> Image.Image:
    if record.path is not None:
        with Image.open(record.path) as image:
            return image.convert("RGB")
    return ensure_pil_rgb(record.image)


def ensure_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, list):
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image, mode="L").convert("RGB")
        if image.ndim == 3:
            return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize((232, 232)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
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


def build_model(num_classes: int, dropout: float, device: torch.device) -> nn.Module:
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def freeze_backbone_train_head(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, unfreeze_from_block: int) -> int:
    freeze_backbone_train_head(model)
    total_blocks = len(model.features)
    block_start = max(0, min(int(unfreeze_from_block), total_blocks - 1))
    for block_index in range(block_start, total_blocks):
        for parameter in model.features[block_index].parameters():
            parameter.requires_grad = True
    return block_start


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def resolve_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def class_index(label: str) -> int:
    normalized = label.strip().lower()
    for index, class_name in enumerate(CLASS_NAMES):
        if class_name.lower() == normalized:
            return index
    raise ValueError(f"Unknown emotion label: {label!r}")


def read_manifest_records(
    dataset_dir: Path,
    personal_weight: float,
    source_name: str = "personal",
) -> list[ImageRecord]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return []

    records = []
    seen_paths: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_text = row.get("label") or ""
            if not label_text and row.get("label_index"):
                label_text = CLASS_NAMES[int(row["label_index"])]
            if not label_text:
                continue

            image_path_text = row.get("image_path") or ""
            if not image_path_text:
                continue

            image_path = Path(image_path_text)
            if not image_path.is_absolute():
                image_path = dataset_dir / image_path
            image_path = image_path.resolve()

            if image_path in seen_paths or not image_path.exists():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            records.append(
                ImageRecord(
                    label=class_index(label_text),
                    source=source_name,
                    weight=personal_weight,
                    path=image_path,
                )
            )
            seen_paths.add(image_path)
    return records


def scan_folder_records(
    dataset_dir: Path,
    personal_weight: float,
    source_name: str = "personal",
) -> list[ImageRecord]:
    records = []
    for label_index, label in enumerate(CLASS_NAMES):
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    ImageRecord(
                        label=label_index,
                        source=source_name,
                        weight=personal_weight,
                        path=image_path.resolve(),
                    )
                )
    return records


def load_personal_records(
    personal_root: Path,
    personal_weight: float,
) -> list[ImageRecord]:
    if not personal_root.exists():
        raise FileNotFoundError(f"Personal dataset root not found: {personal_root}")

    records = read_manifest_records(personal_root, personal_weight=personal_weight)
    if not records:
        records = scan_folder_records(personal_root, personal_weight=personal_weight)
    if records:
        return records

    legacy_records = []
    for legacy_dir in sorted(path for path in personal_root.iterdir() if path.is_dir()):
        if legacy_dir.name in CLASS_NAMES:
            continue
        legacy_records_for_dir = read_manifest_records(
            legacy_dir,
            personal_weight=personal_weight,
            source_name=f"personal:{legacy_dir.name}",
        )
        if not legacy_records_for_dir:
            legacy_records_for_dir = scan_folder_records(
                legacy_dir,
                personal_weight=personal_weight,
                source_name=f"personal:{legacy_dir.name}",
            )
        legacy_records.extend(legacy_records_for_dir)

    if not legacy_records:
        raise RuntimeError(f"No personal face images found in {personal_root}")
    return legacy_records


def split_personal_records(
    records: list[ImageRecord],
    val_fraction: float,
    min_per_class: int,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    groups: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.label].append(record)

    missing = [
        f"{CLASS_NAMES[label]}={len(groups[label])}"
        for label in range(len(CLASS_NAMES))
        if len(groups[label]) < min_per_class
    ]
    if missing:
        raise RuntimeError(
            "Not enough personal images per class. "
            f"Minimum is {min_per_class}; current counts: {', '.join(missing)}"
        )

    rng = random.Random(seed)
    train_records = []
    val_records = []
    for label in range(len(CLASS_NAMES)):
        label_records = list(groups[label])
        if not label_records:
            continue
        rng.shuffle(label_records)
        val_count = max(1, int(round(len(label_records) * val_fraction)))
        val_count = min(val_count, len(label_records) - 1) if len(label_records) > 1 else 0
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])

    if not train_records or not val_records:
        raise RuntimeError("Personal train/validation split produced an empty split.")
    return train_records, val_records


def find_fer_arrow(cache_root: Path, split_name: str) -> Path:
    matches = sorted(cache_root.rglob(f"fer2013-enhanced-{split_name}.arrow"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find fer2013-enhanced-{split_name}.arrow under {cache_root}"
        )
    return matches[0]


def load_fer_replay_records(
    cache_root: Path,
    replay_per_class: int,
    seed: int,
    fer_weight: float,
) -> list[ImageRecord]:
    if replay_per_class <= 0:
        return []

    try:
        from datasets import Dataset as HFDataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'datasets' package is required when --fer-replay-per-class is greater than 0."
        ) from exc

    arrow_path = find_fer_arrow(cache_root, "train")
    fer_dataset = HFDataset.from_file(str(arrow_path))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(fer_dataset["emotion"]):
        label_index = int(label)
        if 0 <= label_index < len(CLASS_NAMES):
            groups[label_index].append(index)

    rng = random.Random(seed)
    records = []
    for label in range(len(CLASS_NAMES)):
        indices = list(groups[label])
        if not indices:
            raise RuntimeError(f"FER replay split has no samples for {CLASS_NAMES[label]}.")
        rng.shuffle(indices)
        selected = indices[: min(replay_per_class, len(indices))]
        for index in selected:
            sample = fer_dataset[int(index)]
            records.append(
                ImageRecord(
                    label=label,
                    source="fer2013_replay",
                    weight=fer_weight,
                    image=ensure_pil_rgb(sample["image"]),
                )
            )
    return records


def build_loader(
    records: list[ImageRecord],
    transform,
    batch_size: int,
    num_workers: int,
    weighted: bool,
) -> DataLoader:
    dataset = EmotionImageDataset(records, transform=transform)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if weighted:
        weights = torch.DoubleTensor([record.weight for record in records])
        sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True)
        return DataLoader(dataset, sampler=sampler, shuffle=False, **loader_kwargs)

    return DataLoader(dataset, shuffle=False, **loader_kwargs)


def compute_metrics(
    predictions: list[int],
    labels: list[int],
    num_classes: int,
) -> dict[str, Any]:
    if not labels:
        return {"acc": 0.0, "macro_f1": 0.0, "per_class_f1": [0.0] * num_classes}

    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    per_class_f1 = []
    for label in range(num_classes):
        tp = sum(1 for pred, actual in zip(predictions, labels) if pred == label and actual == label)
        fp = sum(1 for pred, actual in zip(predictions, labels) if pred == label and actual != label)
        fn = sum(1 for pred, actual in zip(predictions, labels) if pred != label and actual == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_f1.append(f1)

    return {
        "acc": correct / len(labels),
        "macro_f1": sum(per_class_f1) / num_classes,
        "per_class_f1": per_class_f1,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_count = 0
    predictions: list[int] = []
    labels_seen: list[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
                optimizer.step()

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels_seen.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(predictions, labels_seen, num_classes=len(CLASS_NAMES))
    metrics["loss"] = total_loss / max(1, total_count)
    return metrics


def state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    run_stamp: str,
    phase_name: str,
    epoch: int,
    global_step: int,
    val_metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    f1_text = f"{val_metrics['macro_f1']:.4f}".replace(".", "p")
    checkpoint_path = output_dir / f"{run_stamp}_personal_finetune_{phase_name}_f1{f1_text}.pt"
    checkpoint = {
        "epoch": epoch,
        "phase_name": phase_name,
        "model_state_dict": state_dict_on_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "best_val_f1": float(val_metrics["macro_f1"]),
        "best_val_loss": float(val_metrics["loss"]),
        "best_val_acc": float(val_metrics["acc"]),
        "global_step": global_step,
        "config": dict(config),
        "checkpoint_path": str(checkpoint_path),
        "wandb_run_id": None,
        "wandb_run_name": config["run_name"],
    }
    torch.save(checkpoint, checkpoint_path)
    shutil.copyfile(checkpoint_path, output_dir / "latest_personal_finetune.pt")
    return checkpoint_path


def train_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    run_stamp: str,
    config: dict[str, Any],
    best_state: dict[str, Any],
    global_step: int,
) -> tuple[dict[str, Any], int]:
    for epoch in range(1, epochs + 1):
        started_at = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        global_step += len(train_loader)

        elapsed = time.perf_counter() - started_at
        print(
            f"{stage_name} [{epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["macro_f1"] > best_state["best_val_f1"]:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                run_stamp=run_stamp,
                phase_name=stage_name,
                epoch=epoch,
                global_step=global_step,
                val_metrics=val_metrics,
                config=config,
            )
            best_state = {
                "best_val_f1": float(val_metrics["macro_f1"]),
                "checkpoint_path": checkpoint_path,
                "phase_name": stage_name,
                "epoch": epoch,
            }
            print(f"Saved new best checkpoint: {checkpoint_path}")

    return best_state, global_step


def count_by_label(records: list[ImageRecord]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))}


def count_by_source(records: list[ImageRecord]) -> dict[str, int]:
    return dict(Counter(record.source for record in records))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    class_names = list(config.get("class_names") or CLASS_NAMES)
    if class_names != CLASS_NAMES:
        raise RuntimeError(
            "Base checkpoint class order does not match backend class order. "
            f"Expected {CLASS_NAMES}, got {class_names}."
        )

    num_classes = int(config.get("num_classes", len(CLASS_NAMES)))
    if num_classes != len(CLASS_NAMES):
        raise RuntimeError(f"Expected {len(CLASS_NAMES)} classes, got {num_classes}.")

    dropout = float(config.get("dropout", 0.2))
    model = build_model(num_classes=num_classes, dropout=dropout, device=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint, config


def build_training_config(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    personal_records: list[ImageRecord],
    fer_records: list[ImageRecord],
) -> dict[str, Any]:
    config = dict(base_config)
    config.update(
        {
            "strategy": "personal_finetune",
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "run_name": f"personal_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "base_checkpoint": str(args.base_checkpoint),
            "personal_root": str(args.personal_root),
            "fer_cache_root": str(args.fer_cache_root),
            "fer_replay_per_class": args.fer_replay_per_class,
            "personal_sample_weight": args.personal_sample_weight,
            "fer_sample_weight": args.fer_sample_weight,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "min_personal_per_class": args.min_personal_per_class,
            "freeze_epochs": args.epochs_head,
            "finetune_epochs": args.epochs_finetune,
            "lr": args.head_lr,
            "finetune_lr": args.finetune_lr,
            "weight_decay": args.weight_decay,
            "finetune_weight_decay": args.weight_decay,
            "unfreeze_from_block": args.unfreeze_from_block,
            "personal_total": len(personal_records),
            "fer_replay_total": len(fer_records),
            "train_total": len(train_records),
            "val_total": len(val_records),
            "personal_counts": count_by_label(personal_records),
            "train_counts": count_by_label(train_records),
            "val_counts": count_by_label(val_records),
            "train_sources": count_by_source(train_records),
        }
    )
    return config


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Fine-tune the FER classifier on a personal face dataset."
    )
    parser.add_argument("--personal-root", type=Path, default=DEFAULT_PERSONAL_ROOT)
    parser.add_argument("--base-checkpoint", type=Path, default=Path(settings.emotion_model_path))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fer-cache-root", type=Path, default=DEFAULT_FER_CACHE_ROOT)
    parser.add_argument("--fer-replay-per-class", type=int, default=150)
    parser.add_argument("--min-personal-per-class", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--personal-sample-weight", type=float, default=6.0)
    parser.add_argument("--fer-sample-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-from-block", type=int, default=12)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize datasets, then exit before model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.personal_root = resolve_path(args.personal_root)
    args.base_checkpoint = resolve_path(args.base_checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    args.fer_cache_root = resolve_path(args.fer_cache_root)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    personal_records = load_personal_records(
        personal_root=args.personal_root,
        personal_weight=args.personal_sample_weight,
    )
    personal_train_records, val_records = split_personal_records(
        personal_records,
        val_fraction=args.val_fraction,
        min_per_class=args.min_personal_per_class,
        seed=args.seed,
    )
    fer_records = load_fer_replay_records(
        cache_root=args.fer_cache_root,
        replay_per_class=args.fer_replay_per_class,
        seed=args.seed,
        fer_weight=args.fer_sample_weight,
    )
    train_records = personal_train_records + fer_records

    print("Personal counts:", count_by_label(personal_records))
    print("Personal train counts:", count_by_label(personal_train_records))
    print("Personal validation counts:", count_by_label(val_records))
    print("FER replay counts:", count_by_label(fer_records))
    print("Train sources:", count_by_source(train_records))

    if args.dry_run:
        print("Dry run complete. No training was started.")
        return

    model, _base_checkpoint, base_config = load_base_model(args.base_checkpoint, device=device)
    training_config = build_training_config(
        args=args,
        base_config=base_config,
        train_records=train_records,
        val_records=val_records,
        personal_records=personal_records,
        fer_records=fer_records,
    )

    train_transform, eval_transform = build_transforms()
    train_loader = build_loader(
        train_records,
        transform=train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted=True,
    )
    val_loader = build_loader(
        val_records,
        transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weighted=False,
    )
    criterion = nn.CrossEntropyLoss().to(device)

    best_state: dict[str, Any] = {"best_val_f1": -1.0, "checkpoint_path": None}
    global_step = 0
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.epochs_head > 0:
        freeze_backbone_train_head(model)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        print(
            f"Head warmup: epochs={args.epochs_head}, "
            f"trainable_params={count_trainable_parameters(model)}"
        )
        best_state, global_step = train_stage(
            stage_name="head",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs_head,
            output_dir=args.output_dir,
            run_stamp=run_stamp,
            config=training_config,
            best_state=best_state,
            global_step=global_step,
        )

    if args.epochs_finetune > 0:
        block_start = unfreeze_last_blocks(model, args.unfreeze_from_block)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
        )
        print(
            f"Light fine-tune: epochs={args.epochs_finetune}, "
            f"unfreeze_from_block={block_start}, "
            f"trainable_params={count_trainable_parameters(model)}"
        )
        best_state, global_step = train_stage(
            stage_name="finetune",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs_finetune,
            output_dir=args.output_dir,
            run_stamp=run_stamp,
            config=training_config,
            best_state=best_state,
            global_step=global_step,
        )

    if best_state["checkpoint_path"] is None:
        raise RuntimeError("No checkpoint was saved. Run at least one training epoch.")

    checkpoint_path = best_state["checkpoint_path"]
    print("Best checkpoint:")
    print(f"  path={checkpoint_path}")
    print(f"  val_f1={best_state['best_val_f1']:.4f}")
    print("Use it with:")
    print(
        f"  $env:EMOTION_MODEL_PATH='{checkpoint_path}'; "
        "python -m uvicorn assistant_backend.main:app --host 0.0.0.0 --port 8000"
    )


if __name__ == "__main__":
    main()
