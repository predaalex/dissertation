from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

KEY_TO_LABEL = {str(index + 1): label for index, label in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "datasets" / "personal_fer"
MANIFEST_FIELDS = [
    "image_path",
    "label",
    "label_index",
    "timestamp_utc",
    "camera_index",
    "frame_width",
    "frame_height",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
    "saved_from",
]


@dataclass
class CollectionState:
    manifest_path: Path
    manifest_rows: list[dict[str, str]]
    counts: Counter
    saved_stack: list[tuple[Path, dict[str, str]]]


def build_face_detector() -> cv2.CascadeClassifier:
    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}.")
    return detector


def pad_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)
    return x1, y1, x2, y2


def detect_largest_face(
    frame: np.ndarray,
    face_detector: cv2.CascadeClassifier,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    x1, y1, x2, y2 = pad_bbox(x, y, w, h, frame.shape[1], frame.shape[0])
    crop = frame[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def build_no_face_placeholder() -> np.ndarray:
    placeholder = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "No face detected",
        (25, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def count_images(dataset_dir: Path) -> Counter:
    counts: Counter = Counter()
    for label in CLASS_NAMES:
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue
        counts[label] = sum(
            1
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {field: row.get(field, "") for field in MANIFEST_FIELDS}
            for row in csv.DictReader(handle)
        ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def init_collection_state(output_root: Path) -> tuple[Path, CollectionState]:
    dataset_dir = output_root
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for label in CLASS_NAMES:
        (dataset_dir / label).mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "manifest.csv"
    manifest_rows = read_manifest(manifest_path)
    counts = count_images(dataset_dir)

    return dataset_dir, CollectionState(
        manifest_path=manifest_path,
        manifest_rows=manifest_rows,
        counts=counts,
        saved_stack=[],
    )


def print_existing_counts(counts: Counter) -> None:
    existing_total = sum(counts[label] for label in CLASS_NAMES)
    if existing_total == 0:
        print("No existing images found in the personal dataset.")
        return

    print("Existing images found in the personal dataset:")
    for label in CLASS_NAMES:
        print(f"  {label}: {counts[label]}")
    print(f"  total: {existing_total}")


def draw_status_overlay(
    frame: np.ndarray,
    counts: Counter,
    target_per_class: int,
    has_face: bool,
) -> np.ndarray:
    output = frame.copy()
    status_color = (0, 255, 0) if has_face else (0, 0, 255)
    cv2.putText(
        output,
        "Face detected" if has_face else "No face detected - label save is blocked",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    y = 70
    for index, label in enumerate(CLASS_NAMES, start=1):
        text = f"{index}: {label:<8} {counts[label]:>4}/{target_per_class}"
        color = (0, 220, 0) if counts[label] >= target_per_class else (255, 255, 255)
        cv2.putText(
            output,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28

    footer = "Press 1-7 to save label, u to undo, q to quit"
    cv2.putText(
        output,
        footer,
        (20, output.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def save_crop(
    crop: np.ndarray,
    label: str,
    bbox: tuple[int, int, int, int],
    frame: np.ndarray,
    dataset_dir: Path,
    camera_index: int,
    jpeg_quality: int,
    manifest_rows: list[dict[str, str]],
) -> tuple[Path, dict[str, str]]:
    label_dir = dataset_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = utc_stamp()
    label_count = sum(1 for row in manifest_rows if row.get("label") == label) + 1
    file_name = f"{timestamp}_{label.lower()}_{label_count:04d}.jpg"
    image_path = label_dir / file_name

    success = cv2.imwrite(str(image_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise RuntimeError(f"Failed to write image to {image_path}.")

    x1, y1, x2, y2 = bbox
    relative_path = image_path.relative_to(dataset_dir).as_posix()
    row = {
        "image_path": relative_path,
        "label": label,
        "label_index": str(CLASS_NAMES.index(label)),
        "timestamp_utc": timestamp,
        "camera_index": str(camera_index),
        "frame_width": str(frame.shape[1]),
        "frame_height": str(frame.shape[0]),
        "crop_x1": str(x1),
        "crop_y1": str(y1),
        "crop_x2": str(x2),
        "crop_y2": str(y2),
        "crop_width": str(crop.shape[1]),
        "crop_height": str(crop.shape[0]),
        "saved_from": "live_camera_face_crop",
    }
    manifest_rows.append(row)
    return image_path, row


def undo_last_save(
    saved_stack: list[tuple[Path, dict[str, str]]],
    manifest_rows: list[dict[str, str]],
    counts: Counter,
) -> bool:
    if not saved_stack:
        return False

    image_path, row = saved_stack.pop()
    if image_path.exists():
        image_path.unlink()

    for index in range(len(manifest_rows) - 1, -1, -1):
        if manifest_rows[index] == row:
            del manifest_rows[index]
            break

    label = row["label"]
    counts[label] = max(0, counts[label] - 1)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a personal facial emotion dataset from the webcam."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir, state = init_collection_state(args.output_root)

    face_detector = build_face_detector()
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    print(f"Collecting personal FER dataset in: {dataset_dir}")
    print_existing_counts(state.counts)
    print("Class keys:")
    for key, label in KEY_TO_LABEL.items():
        print(f"  {key} -> {label}")
    print("Controls: 1-7 save current face crop, u undo, q quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            display_frame = frame.copy()
            face_crop, face_bbox = detect_largest_face(frame, face_detector)
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            overlay = draw_status_overlay(
                display_frame,
                counts=state.counts,
                target_per_class=args.target_per_class,
                has_face=face_crop is not None,
            )
            cv2.imshow("Personal FER Dataset Collector", overlay)

            if face_crop is None or face_crop.size == 0:
                cv2.imshow("Current Face Crop", build_no_face_placeholder())
            else:
                cv2.imshow("Current Face Crop", cv2.resize(face_crop, (320, 320)))

            key_code = cv2.waitKey(1) & 0xFF
            if key_code == 255:
                continue

            key = chr(key_code).lower()
            if key == "q":
                break

            if key == "u":
                if undo_last_save(state.saved_stack, state.manifest_rows, state.counts):
                    write_manifest(state.manifest_path, state.manifest_rows)
                    print("Undid last saved image.")
                else:
                    print("Nothing to undo in this collector run.")
                continue

            if key not in KEY_TO_LABEL:
                continue

            label = KEY_TO_LABEL[key]
            if face_crop is None or face_bbox is None or face_crop.size == 0:
                print(f"Skipped {label}: no face detected.")
                continue

            image_path, row = save_crop(
                crop=face_crop,
                label=label,
                bbox=face_bbox,
                frame=frame,
                dataset_dir=dataset_dir,
                camera_index=args.camera_index,
                jpeg_quality=args.jpeg_quality,
                manifest_rows=state.manifest_rows,
            )
            write_manifest(state.manifest_path, state.manifest_rows)
            state.saved_stack.append((image_path, row))
            state.counts[label] += 1
            print(f"Saved {label}: {image_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        write_manifest(state.manifest_path, state.manifest_rows)

    print("Final class counts:")
    for label in CLASS_NAMES:
        print(f"  {label}: {state.counts[label]}")
    print(f"Manifest: {state.manifest_path}")


if __name__ == "__main__":
    main()
