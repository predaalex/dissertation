import argparse

import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument(
        "--disable-face-cropping",
        action="store_true",
        help="Disable backend face detection/cropping for this request.",
    )
    args = parser.parse_args()

    with open(args.image_path, "rb") as image_file:
        files = {
            "image": (args.image_path, image_file, "image/jpeg"),
        }
        response = requests.post(
            f"{BASE_URL}/emotion/predict",
            files=files,
            data={"disable_face_cropping": str(args.disable_face_cropping).lower()},
            timeout=60,
        )

    print("Status code:", response.status_code)
    try:
        print(response.json())
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
