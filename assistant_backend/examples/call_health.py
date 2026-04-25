import requests


BASE_URL = "http://127.0.0.1:8000"


def main():
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print("Status code:", response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
