import requests
from pathlib import Path


def download_file(url: str, path: Path):
    response = requests.get(url, stream=True)
    with path.open('wb') as file:
        for data in response.iter_content():
            file.write(data)
