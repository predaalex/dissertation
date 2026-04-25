import argparse
from pathlib import Path
import time

import cv2
import requests


BASE_URL = "http://127.0.0.1:8000"


def encode_frame(frame):
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame as JPEG.")
    return encoded_image.tobytes()


def predict_frame(frame, timeout=60):
    encoded_bytes = encode_frame(frame)
    files = {
        "image": ("camera_frame.jpg", encoded_bytes, "image/jpeg"),
    }
    response = requests.post(
        f"{BASE_URL}/emotion/predict",
        files=files,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def predict_and_cache_frame(frame, session_id, timeout=60):
    encoded_bytes = encode_frame(frame)
    files = {
        "image": ("camera_frame.jpg", encoded_bytes, "image/jpeg"),
    }
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/emotion/predict",
        files=files,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--save-capture-path", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.5,
        help="Seconds between automatic emotion refreshes when --session-id is provided.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Disable automatic polling and only classify when pressing 'c'.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not open the OpenCV preview window.",
    )
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    latest_prediction = None
    last_poll_time = 0.0

    print("Camera started.")
    print("Controls:")
    if args.session_id:
        if args.manual_only:
            print("  c -> capture current frame, predict emotion, and cache it into the session")
        else:
            print(
                f"  automatic polling -> refresh session emotion every {args.poll_interval:.1f}s"
            )
            print("  c -> force an immediate refresh")
    else:
        print("  c -> capture current frame and send it to /emotion/predict")
    if args.no_preview:
        print("  Ctrl+C -> quit")
    else:
        print("  q -> quit")

    def run_prediction(current_frame, triggered_by="manual"):
        nonlocal latest_prediction, last_poll_time
        if args.session_id:
            latest_prediction = predict_and_cache_frame(current_frame, args.session_id)
        else:
            latest_prediction = predict_frame(current_frame)

        last_poll_time = time.time()
        print(f"Prediction ({triggered_by}):", latest_prediction)

        if args.save_capture_path:
            output_path = Path(args.save_capture_path)
            cv2.imwrite(str(output_path), current_frame)
            print(f"Saved captured frame to {output_path}")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            display_frame = frame.copy()

            if latest_prediction is not None:
                vote_count = latest_prediction.get("vote_count")
                window_size = latest_prediction.get("window_size")
                accepted = latest_prediction.get("accepted")
                if vote_count is not None and window_size is not None:
                    overlay = (
                        f"{latest_prediction['emotion_label']} "
                        f"({latest_prediction['confidence']:.2f}) "
                        f"[votes {vote_count}/{window_size}] "
                        f"{'accepted' if accepted is not False else 'rejected'}"
                    )
                else:
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

            if args.session_id and not args.manual_only:
                next_refresh = max(0.0, args.poll_interval - (time.time() - last_poll_time))
                refresh_text = f"Auto refresh in {next_refresh:.1f}s"
            elif args.session_id:
                refresh_text = "Manual refresh mode"
            else:
                refresh_text = "Standalone prediction mode"

            cv2.putText(
                display_frame,
                refresh_text,
                (20, display_frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                "Press 'c' to classify now, 'q' to quit",
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.session_id and not args.manual_only:
                current_time = time.time()
                if current_time - last_poll_time >= args.poll_interval:
                    try:
                        run_prediction(frame, triggered_by="auto")
                    except requests.HTTPError as exc:
                        print(f"Backend returned an error: {exc}")
                        if exc.response is not None:
                            print(exc.response.text)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to classify frame: {exc}")

            if not args.no_preview:
                cv2.imshow("Emotion Camera Feed", display_frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = None
                time.sleep(0.01)

            if key == ord("q"):
                break

            if key == ord("c"):
                try:
                    run_prediction(frame, triggered_by="manual")
                except requests.HTTPError as exc:
                    print(f"Backend returned an error: {exc}")
                    if exc.response is not None:
                        print(exc.response.text)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to classify frame: {exc}")
    finally:
        cam.release()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
