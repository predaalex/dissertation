import argparse
import subprocess
import sys
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"


def create_session():
    response = requests.post(f"{BASE_URL}/sessions", timeout=10)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-capture-path", default=None)
    parser.add_argument(
        "--disable-face-cropping",
        action="store_true",
        help="Disable local and backend face cropping in the camera poller.",
    )
    args = parser.parse_args()

    session = create_session()
    session_id = session["session_id"]

    print("Created session:")
    print(session)
    print(f"Session id: {session_id}")
    print("Starting camera poller...")

    script_path = Path(__file__).resolve().parent / "predict_emotion_from_camera.py"
    command = [
        sys.executable,
        str(script_path),
        "--session-id",
        session_id,
        "--camera-index",
        str(args.camera_index),
        "--poll-interval",
        str(args.poll_interval),
    ]

    if args.manual_only:
        command.append("--manual-only")
    if args.no_preview:
        command.append("--no-preview")
    if args.disable_face_cropping:
        command.append("--disable-face-cropping")
    if args.save_capture_path:
        command.extend(["--save-capture-path", args.save_capture_path])

    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
