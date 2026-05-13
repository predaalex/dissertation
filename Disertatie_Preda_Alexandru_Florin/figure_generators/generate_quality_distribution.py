from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk


DEFAULT_DATASET_NAME = "abhilash88/fer2013-enhanced"
TITLE_FONT_SIZE = 28 * 1.5
LABEL_FONT_SIZE = 22 * 1.5
TICK_FONT_SIZE = 20 * 1.5
ANNOTATION_FONT_SIZE = 18 * 1.5


def default_paths() -> tuple[Path, Path]:
    thesis_dir = Path(__file__).resolve().parents[1]
    project_root = thesis_dir.parent
    cache_dir = project_root / "facial_classifier" / "datasets" / "fer2013-enhanced"
    output_path = thesis_dir / "images" / "fer2013_enhanced_quality_distribution.png"
    return cache_dir, output_path


def parse_args() -> argparse.Namespace:
    cache_dir, output_path = default_paths()
    parser = argparse.ArgumentParser(
        description="Generate a quality score histogram for FER2013 Enhanced."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local dataset saved with datasets.save_to_disk().",
    )
    parser.add_argument("--cache-dir", default=str(cache_dir))
    parser.add_argument("--output-path", default=str(output_path))
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.35, 0.5],
        help="Quality thresholds to mark on the histogram.",
    )
    return parser.parse_args()


def load_splits(args: argparse.Namespace):
    if args.dataset_path:
        return load_from_disk(args.dataset_path)
    return load_dataset(args.dataset_name, cache_dir=args.cache_dir)


def collect_quality_scores(dataset) -> list[float]:
    scores: list[float] = []
    for split_name, split in dataset.items():
        if "quality_score" not in split.column_names:
            raise KeyError(f"Split '{split_name}' does not contain a quality_score column.")
        scores.extend(float(score) for score in split["quality_score"])
    return scores


def plot_quality_distribution(scores: list[float], thresholds: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax.hist(scores, bins=35, color="#55a868", edgecolor="white")

    for threshold in thresholds:
        ax.axvline(threshold, color="#c44e52", linestyle="--", linewidth=1.5)
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.95,
            f"{threshold:g}",
            color="#c44e52",
            ha="right",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            rotation=90,
        )

    ax.set_title("FER2013 Enhanced Quality Score Distribution", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Quality score", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Number of images", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_splits(args)
    scores = collect_quality_scores(dataset)
    plot_quality_distribution(scores, args.thresholds, Path(args.output_path))
    print(f"Saved quality distribution figure to {args.output_path}")


if __name__ == "__main__":
    main()
