import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_dataset

ORIGINAL_CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

TARGET_CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
LABEL_MAP = {
    0: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}


def remap_sample(sample):
    old_label = int(sample["emotion"])
    new_label = LABEL_MAP[old_label]

    sample["original_emotion"] = old_label
    sample["original_emotion_name"] = ORIGINAL_CLASS_NAMES[old_label]
    sample["emotion"] = new_label
    sample["emotion_name"] = TARGET_CLASS_NAMES[new_label]

    return sample


def recompute_sample_weights(split):
    labels = [int(label) for label in split["emotion"]]
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(TARGET_CLASS_NAMES)
    weights = [total / (num_classes * counts[int(label)]) for label in labels]
    return split.add_column("sample_weight", weights)


def build_no_disgust_dataset(dataset_name, cache_dir):
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    filtered = {}

    for split_name, split in dataset.items():
        split = split.filter(lambda sample: int(sample["emotion"]) != 1)
        split = split.map(remap_sample)

        if "sample_weight" in split.column_names:
            split = split.remove_columns("sample_weight")
        split = recompute_sample_weights(split)

        filtered[split_name] = split

    return DatasetDict(filtered)


def write_metadata(output_dir, dataset_name):
    metadata = {
        "dataset_name": "fer2013-enhanced-no-disgust",
        "source_dataset": dataset_name,
        "removed_classes": [{"id": 1, "name": "Disgust"}],
        "label_map": {str(old): new for old, new in LABEL_MAP.items()},
        "class_names": TARGET_CLASS_NAMES,
        "num_classes": len(TARGET_CLASS_NAMES),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a local FER2013-enhanced variant without the Disgust class."
    )
    parser.add_argument("--dataset-name", default="abhilash88/fer2013-enhanced")
    parser.add_argument("--cache-dir", default="datasets/fer2013-enhanced")
    parser.add_argument("--output-dir", default="datasets/fer2013-enhanced-no-disgust")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_no_disgust_dataset(args.dataset_name, args.cache_dir)
    dataset.save_to_disk(str(output_dir))
    write_metadata(output_dir, args.dataset_name)

    print(f"Saved no-Disgust dataset to {output_dir}")
    for split_name, split in dataset.items():
        counts = Counter(int(label) for label in split["emotion"])
        print(f"{split_name}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
