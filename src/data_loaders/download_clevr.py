import os
import hashlib
import requests
import tqdm
from zipfile import ZipFile

class CLEVRDownloader:
    CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    CLEVR_MD5 = "b11922020e72d0cd9154779b2d3d07d2"
    ZIP_NAME = "CLEVR_v1.0.zip"
    EXTRACTED_DIR = "CLEVR_v1.0"

    def __init__(self, root_dir="data/clevr"):
        self.root_dir = root_dir
        self.zip_path = os.path.join(root_dir, self.ZIP_NAME)
        self.extracted_path = os.path.join(root_dir, self.EXTRACTED_DIR)
        os.makedirs(root_dir, exist_ok=True)

    @staticmethod
    def md5(fname, chunk_size=8192):
        h = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def download(self):
        if os.path.exists(self.zip_path):
            print("CLEVR ZIP already exists. Skipping download.")
            return

        print("Downloading CLEVR dataset...")
        try:
            resp = requests.get(self.CLEVR_URL, stream=True, timeout=10)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(self.zip_path, "wb") as f, tqdm.tqdm(
                desc="Downloading CLEVR",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Download failed: {e}")

    def verify(self):
        print("Verifying checksum...")
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError("CLEVR ZIP not found.")
        if self.md5(self.zip_path) != self.CLEVR_MD5:
            raise ValueError("MD5 checksum does not match. Delete ZIP and retry.")
        print("Checksum OK.")

    def extract(self):
        if os.path.exists(self.extracted_path):
            print("CLEVR already extracted. Skipping extraction.")
            return

        print("Extracting CLEVR dataset...")
        with ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.root_dir)
        print(f"Extraction complete. CLEVR ready at {self.extracted_path}")

    def setup(self):
        self.download()
        self.verify()
        self.extract()

if __name__ == "__main__":
    clevr = CLEVRDownloader(root_dir="data/clevr")
    clevr.setup()
