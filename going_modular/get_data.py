
import os
import requests
import zipfile
from pathlib import Path

def data_access(download_path: Path, download_url: str):
    if download_path.is_dir():
        print(f"Data already exists at {Path(download_path)}, skipping download.")
    else:
        print("Not existing. Creating one.")
        download_path.mkdir(parents = True, exist_ok = True)

    with open(download_path/"pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(download_url)
        print("Downloading...")
        f.write(request.content)

    with zipfile.ZipFile(download_path/"pizza_steak_sushi.zip", "r") as zipf:
        print("Unzipping...")
        zipf.extractall(download_path)

    os.remove(download_path/"pizza_steak_sushi.zip")
    print("Successfully downloaded the data.")

if __name__ == "__main__":
    download_path = Path("data/pizza_steak_sushi/")
    download_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    data_access(download_path=download_path, download_url=download_url)
