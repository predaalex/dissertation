import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", required=True)
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task to run. Use translate to translate speech into English.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Shortcut for --task translate.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional Whisper language code, for example en.",
    )
    args = parser.parse_args()
    task = "translate" if args.translate else args.task

    with open(args.audio_path, "rb") as audio_file:
        files = {
            "audio": (args.audio_path, audio_file, "audio/wav"),
        }
        response = requests.post(
            f"{BASE_URL}/speech-to-text/transcribe",
            files=files,
            data={"task": task, "language": args.language or ""},
            timeout=300,
        )

    print("Status code:", response.status_code)
    try:
        print(response.json())
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
