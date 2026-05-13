from __future__ import annotations

from pathlib import Path

from datasets import Dataset


DATASET_CACHE_ROOT = Path("datasets/fer2013-enhanced")
DATASET_CACHE_HASH = Path(
    "abhilash88___fer2013-enhanced/default/0.0.0/"
    "9daf28bf19b0aa787ea976e1d9ba1f98c4b9681b"
)
SPLITS = ("train", "validation", "test")
QUALITY_THRESHOLDS = (0.0, 0.35, 0.5)


def load_cached_split(split_name: str) -> Dataset:
    arrow_path = (
        DATASET_CACHE_ROOT
        / DATASET_CACHE_HASH
        / f"fer2013-enhanced-{split_name}.arrow"
    )
    return Dataset.from_file(str(arrow_path))


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def main() -> None:
    dataset = {split_name: load_cached_split(split_name) for split_name in SPLITS}
    total_samples = sum(len(split) for split in dataset.values())

    print("Dataset splits")
    print("split,samples,percentage")
    for split_name, split in dataset.items():
        percentage = 100.0 * len(split) / total_samples
        print(f"{split_name},{len(split)},{format_pct(percentage)}")
    print(f"total,{total_samples},100.0%")

    print()
    print("Quality threshold retention")
    print("threshold,train,validation,test,total,percentage")
    for threshold in QUALITY_THRESHOLDS:
        counts = {
            split_name: sum(
                1 for score in split["quality_score"] if float(score) >= threshold
            )
            for split_name, split in dataset.items()
        }
        kept_total = sum(counts.values())
        kept_percentage = 100.0 * kept_total / total_samples
        print(
            f"{threshold:.2f},"
            f"{counts['train']},"
            f"{counts['validation']},"
            f"{counts['test']},"
            f"{kept_total},"
            f"{format_pct(kept_percentage)}"
        )


if __name__ == "__main__":
    main()
