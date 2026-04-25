import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    args = parser.parse_args()

    with open(args.image_path, "rb") as image_file:
        files = {
            "image": (args.image_path, image_file, "image/jpeg"),
        }
        response = requests.post(
            f"{BASE_URL}/emotion/predict",
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
