import sys
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
        
    drive_id = '1FiK0hcP1pz_sXcBRLYsGLBNCkir0od2t'
    size = '460MB'
    print(f"dowload trained model for Bejing with {size}")
    download_file_from_google_drive(drive_id, './Beijing.zip')
    print('done')


if __name__ == "__main__":
    main()