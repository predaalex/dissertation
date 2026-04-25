import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"


def create_session():
    response = requests.post(f"{BASE_URL}/sessions", timeout=10)
    response.raise_for_status()
    return response.json()


def start_camera_poller(
    session_id,
    camera_index,
    poll_interval,
    no_preview,
    manual_only,
    show_camera_logs,
):
    script_path = Path(__file__).resolve().parent / "predict_emotion_from_camera.py"
    command = [
        sys.executable,
        str(script_path),
        "--session-id",
        session_id,
        "--camera-index",
        str(camera_index),
        "--poll-interval",
        str(poll_interval),
    ]

    if no_preview:
        command.append("--no-preview")
    if manual_only:
        command.append("--manual-only")

    stdout = None if show_camera_logs else subprocess.DEVNULL
    stderr = None if show_camera_logs else subprocess.DEVNULL
    return subprocess.Popen(command, stdout=stdout, stderr=stderr)


def stream_assistant_reply(session_id, prompt, use_cached_emotion=True):
    data = {
        "session_id": session_id,
        "text": prompt,
        "use_cached_emotion": str(use_cached_emotion).lower(),
    }

    with requests.post(
        f"{BASE_URL}/assistant/respond",
        data=data,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        metadata = None
        done_payload = None
        current_event = None

        print("\nassistant> ", end="", flush=True)

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue

            if not line.startswith("data: "):
                continue

            payload = json.loads(line[len("data: ") :])

            if current_event == "metadata":
                metadata = payload
            elif current_event == "token":
                token = payload.get("text", "")
                print(token, end="", flush=True)
            elif current_event == "error":
                print(f"\n[error] {payload.get('stage')}: {payload.get('message')}")
                return metadata, None
            elif current_event == "done":
                done_payload = payload

        print()
        return metadata, done_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument(
        "--show-video-preview",
        action="store_true",
        help="Show the OpenCV camera preview window. Disabled by default.",
    )
    parser.add_argument(
        "--show-camera-logs",
        action="store_true",
        help="Show background camera poller logs in the terminal. Disabled by default.",
    )
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after starting the camera poller before the first chat prompt.",
    )
    args = parser.parse_args()

    session = create_session()
    session_id = session["session_id"]

    print("Created session:")
    print(session)
    print(f"\nSession id: {session_id}")
    print("Starting camera emotion poller...")

    camera_process = start_camera_poller(
        session_id=session_id,
        camera_index=args.camera_index,
        poll_interval=args.poll_interval,
        no_preview=not args.show_video_preview,
        manual_only=args.manual_only,
        show_camera_logs=args.show_camera_logs,
    )

    print(
        f"Waiting {args.startup_wait:.1f}s for initial emotion polling before chat starts..."
    )
    time.sleep(args.startup_wait)

    print("\nInteractive chat started.")
    print("Type your message and press Enter.")
    print("Commands:")
    print("  /session  -> show session id")
    print("  /state    -> fetch current session state")
    print("  /quit     -> stop chat and camera poller")

    try:
        while True:
            user_text = input("\nyou> ").strip()

            if not user_text:
                continue

            if user_text == "/quit":
                break

            if user_text == "/session":
                print(f"session_id={session_id}")
                continue

            if user_text == "/state":
                response = requests.get(f"{BASE_URL}/sessions/{session_id}", timeout=10)
                response.raise_for_status()
                print(response.json())
                continue

            metadata, done_payload = stream_assistant_reply(
                session_id=session_id,
                prompt=user_text,
                use_cached_emotion=True,
            )

            if metadata is not None:
                print(
                    f"[emotion used] label={metadata.get('emotion_label')} "
                    f"confidence={metadata.get('emotion_confidence')}"
                )

            if done_payload is not None:
                print(
                    f"[latency] {done_payload.get('latency_ms')} ms"
                )
    finally:
        if camera_process.poll() is None:
            camera_process.terminate()
            try:
                camera_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                camera_process.kill()


if __name__ == "__main__":
    main()
