from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk


DEFAULT_DATASET_NAME = "abhilash88/fer2013-enhanced"
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
TITLE_FONT_SIZE = 28 * 1.5
LABEL_FONT_SIZE = 22 * 1.5
TICK_FONT_SIZE = 20 * 1.5
ANNOTATION_FONT_SIZE = 18 * 1.5


def default_paths() -> tuple[Path, Path]:
    thesis_dir = Path(__file__).resolve().parents[1]
    project_root = thesis_dir.parent
    cache_dir = project_root / "facial_classifier" / "datasets" / "fer2013-enhanced"
    output_path = thesis_dir / "images" / "fer2013_enhanced_class_distribution.png"
    return cache_dir, output_path


def parse_args() -> argparse.Namespace:
    cache_dir, output_path = default_paths()
    parser = argparse.ArgumentParser(
        description="Generate a class distribution bar chart for FER2013 Enhanced."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local dataset saved with datasets.save_to_disk().",
    )
    parser.add_argument("--cache-dir", default=str(cache_dir))
    parser.add_argument("--output-path", default=str(output_path))
    return parser.parse_args()


def load_splits(args: argparse.Namespace):
    if args.dataset_path:
        return load_from_disk(args.dataset_path)
    return load_dataset(args.dataset_name, cache_dir=args.cache_dir)


def resolve_class_names(dataset) -> list[str]:
    first_split = next(iter(dataset.values()))
    feature = first_split.features.get("emotion")
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    return CLASS_NAMES


def count_labels(dataset) -> tuple[list[str], list[int]]:
    class_names = resolve_class_names(dataset)
    counts = Counter()
    for split in dataset.values():
        counts.update(int(label) for label in split["emotion"])
    return class_names, [counts.get(index, 0) for index in range(len(class_names))]


def plot_class_distribution(class_names: list[str], counts: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    bars = ax.bar(class_names, counts, color="#4f77b3")
    ax.set_ylim(0, max(counts) * 1.18)

    ax.set_title("FER2013 Enhanced Class Distribution", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Emotion class", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Number of images", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_splits(args)
    class_names, counts = count_labels(dataset)
    plot_class_distribution(class_names, counts, Path(args.output_path))
    print(f"Saved class distribution figure to {args.output_path}")


if __name__ == "__main__":
    main()
