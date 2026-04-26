import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import requests


BASE_URL = "http://127.0.0.1:8000"


def build_face_detector():
    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}.")
    return detector


def detect_largest_face(frame, face_detector):
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
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def pad_bbox(x, y, w, h, frame_width, frame_height):
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)
    return x1, y1, x2, y2


def build_no_face_placeholder():
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


def encode_frame(frame):
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame as JPEG.")
    return encoded_image.tobytes()


def predict_frame(frame, timeout=60, disable_face_cropping=False):
    encoded_bytes = encode_frame(frame)
    files = {
        "image": ("camera_frame.jpg", encoded_bytes, "image/jpeg"),
    }
    response = requests.post(
        f"{BASE_URL}/emotion/predict",
        files=files,
        data={"disable_face_cropping": str(disable_face_cropping).lower()},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def predict_and_cache_frame(frame, session_id, timeout=60, disable_face_cropping=False):
    encoded_bytes = encode_frame(frame)
    files = {
        "image": ("camera_frame.jpg", encoded_bytes, "image/jpeg"),
    }
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/emotion/predict",
        files=files,
        data={"disable_face_cropping": str(disable_face_cropping).lower()},
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
    parser.add_argument(
        "--disable-face-cropping",
        action="store_true",
        help="Disable local and backend face cropping; classify the full frame instead.",
    )
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    face_detector = None
    if not args.disable_face_cropping:
        face_detector = build_face_detector()

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
    if args.disable_face_cropping:
        print("  face cropping -> disabled")
    else:
        print("  face cropping -> enabled (largest detected face)")
    if args.no_preview:
        print("  Ctrl+C -> quit")
    else:
        print("  q -> quit")

    def run_prediction(current_frame, current_face_crop=None, triggered_by="manual"):
        nonlocal latest_prediction, last_poll_time
        request_frame = current_frame
        disable_backend_face_cropping = args.disable_face_cropping

        if not args.disable_face_cropping and current_face_crop is not None:
            request_frame = current_face_crop
            disable_backend_face_cropping = True

        if args.session_id:
            latest_prediction = predict_and_cache_frame(
                request_frame,
                args.session_id,
                disable_face_cropping=disable_backend_face_cropping,
            )
        else:
            latest_prediction = predict_frame(
                request_frame,
                disable_face_cropping=disable_backend_face_cropping,
            )

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
            preview_face_crop = None
            preview_face_bbox = None

            if face_detector is not None:
                preview_face_crop, preview_face_bbox = detect_largest_face(frame, face_detector)
                if preview_face_bbox is not None:
                    x1, y1, x2, y2 = preview_face_bbox
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

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
                        run_prediction(frame, current_face_crop=preview_face_crop, triggered_by="auto")
                    except requests.HTTPError as exc:
                        print(f"Backend returned an error: {exc}")
                        if exc.response is not None:
                            print(exc.response.text)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to classify frame: {exc}")

            if not args.no_preview:
                cv2.imshow("Emotion Camera Feed", display_frame)
                if not args.disable_face_cropping:
                    if preview_face_crop is not None and preview_face_crop.size > 0:
                        crop_preview = cv2.resize(preview_face_crop, (320, 320))
                    else:
                        crop_preview = build_no_face_placeholder()
                    cv2.imshow("Detected Face Crop", crop_preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = None
                time.sleep(0.01)

            if key == ord("q"):
                break

            if key == ord("c"):
                try:
                    run_prediction(frame, current_face_crop=preview_face_crop, triggered_by="manual")
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
