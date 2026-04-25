import argparse
import json

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--use-cached-emotion", default="true")
    args = parser.parse_args()

    data = {
        "session_id": args.session_id,
        "text": args.text,
        "use_cached_emotion": args.use_cached_emotion,
    }

    with requests.post(
        f"{BASE_URL}/assistant/respond",
        data=data,
        stream=True,
        timeout=300,
    ) as response:
        print("Status code:", response.status_code)
        print("--- SSE stream start ---")
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(line)

            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                if "text" in payload:
                    print(payload["text"], end="", flush=True)
        print("\n--- SSE stream end ---")


if __name__ == "__main__":
    main()
