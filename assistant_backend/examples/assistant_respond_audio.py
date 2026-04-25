import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--use-cached-emotion", default="true")
    args = parser.parse_args()

    data = {
        "session_id": args.session_id,
        "use_cached_emotion": args.use_cached_emotion,
    }

    with open(args.audio_path, "rb") as audio_file:
        files = {
            "audio": (args.audio_path, audio_file, "audio/wav"),
        }

        with requests.post(
            f"{BASE_URL}/assistant/respond",
            data=data,
            files=files,
            stream=True,
            timeout=300,
        ) as response:
            print("Status code:", response.status_code)
            print("--- SSE stream start ---")
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    print(line)
            print("--- SSE stream end ---")


if __name__ == "__main__":
    main()
