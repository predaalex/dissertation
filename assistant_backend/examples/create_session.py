import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    response = requests.post(f"{BASE_URL}/sessions", timeout=10)
    print("Status code:", response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
