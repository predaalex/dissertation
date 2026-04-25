import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--confidence", type=float, default=None)
    args = parser.parse_args()

    payload = {
        "emotion_label": args.emotion,
        "confidence": args.confidence,
    }

    response = requests.post(
        f"{BASE_URL}/sessions/{args.session_id}/emotion",
        json=payload,
        timeout=10,
    )
    print("Status code:", response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
