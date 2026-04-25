import argparse
from pathlib import Path

import cv2
import requests


BASE_URL = "http://127.0.0.1:8000"


def predict_frame(frame, timeout=60):
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame as JPEG.")

    files = {
        "image": ("camera_frame.jpg", encoded_image.tobytes(), "image/jpeg"),
    }
    response = requests.post(
        f"{BASE_URL}/emotion/predict",
        files=files,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--save-capture-path", default=None)
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    latest_prediction = None

    print("Camera started.")
    print("Controls:")
    print("  c -> capture current frame and send it to /emotion/predict")
    print("  q -> quit")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            display_frame = frame.copy()

            if latest_prediction is not None:
                overlay = (
                    f"{latest_prediction['emotion_label']} "
                    f"({latest_prediction['confidence']:.2f})"
                )
                cv2.putText(
                    display_frame,
                    overlay,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display_frame,
                "Press 'c' to classify frame, 'q' to quit",
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Emotion Camera Feed", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                try:
                    latest_prediction = predict_frame(frame)
                    print("Prediction:", latest_prediction)

                    if args.save_capture_path:
                        output_path = Path(args.save_capture_path)
                        cv2.imwrite(str(output_path), frame)
                        print(f"Saved captured frame to {output_path}")
                except requests.HTTPError as exc:
                    print(f"Backend returned an error: {exc}")
                    if exc.response is not None:
                        print(exc.response.text)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to classify frame: {exc}")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
