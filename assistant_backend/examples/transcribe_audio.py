import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", required=True)
    args = parser.parse_args()

    with open(args.audio_path, "rb") as audio_file:
        files = {
            "audio": (args.audio_path, audio_file, "audio/wav"),
        }
        response = requests.post(
            f"{BASE_URL}/speech-to-text/transcribe",
            files=files,
            timeout=60,
        )

    print("Status code:", response.status_code)
    try:
        print(response.json())
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
