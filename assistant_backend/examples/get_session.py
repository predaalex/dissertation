import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    response = requests.get(f"{BASE_URL}/sessions/{args.session_id}", timeout=10)
    print("Status code:", response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
