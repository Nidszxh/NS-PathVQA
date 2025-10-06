import os
import hashlib
import requests
import tqdm
from zipfile import ZipFile, BadZipFile

class CLEVRDownloader:
    CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    CLEVR_MD5 = "b11922020e72d0cd9154779b2d3d07d2"
    ZIP_NAME = "CLEVR_v1.0.zip"
    EXTRACTED_DIR = "CLEVR_v1.0"

    def __init__(self, root_dir="data/clevr"):
        self.root_dir = os.path.abspath(root_dir)
        self.zip_path = os.path.join(self.root_dir, self.ZIP_NAME)
        self.extracted_path = os.path.join(self.root_dir, self.EXTRACTED_DIR)
        os.makedirs(self.root_dir, exist_ok=True)

    @staticmethod
    def md5(file_path, chunk_size=1024 * 1024):
        # Compute MD5 checksum efficiently for large files.
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def download(self):
        if os.path.exists(self.zip_path):
            print("CLEVR ZIP already exists. Skipping download.")
            return

        print("Downloading CLEVR dataset...")
        try:
            with requests.get(self.CLEVR_URL, stream=True, timeout=(5, 30)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(self.zip_path, "wb") as f, tqdm(total=total, unit="B",
                unit_scale=True, unit_divisor=1024, desc="Downloading CLEVR",
                    dynamic_ncols=True) as bar:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:  
                            f.write(chunk)
                            bar.update(len(chunk))
        except requests.RequestException as e:
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)
            raise RuntimeError(f"Download failed: {e}")

    def verify(self):
        # Verify file integrity using MD5 checksum.
        print("Verifying checksum...")
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError("CLEVR ZIP not found.")
        file_md5 = self.md5(self.zip_path)
        if file_md5 != self.CLEVR_MD5:
            os.remove(self.zip_path)
            raise ValueError(
                f"MD5 mismatch.\nExpected: {self.CLEVR_MD5}\nFound: {file_md5}\nZIP deleted. Please retry download."
            )
        print("Checksum verified successfully.")

    def extract(self):
        # Extract ZIP file safely.
        if os.path.exists(self.extracted_path):
            print("CLEVR already extracted. Skipping extraction.")
            return

        print("Extracting CLEVR dataset...")
        try:
            with ZipFile(self.zip_path, "r") as zf:
                zf.extractall(self.root_dir)
            print(f"Extraction complete. Dataset ready at: {self.extracted_path}")
        except BadZipFile:
            os.remove(self.zip_path)
            raise RuntimeError("Corrupted ZIP file. Deleted. Please re-download.")

    def setup(self):
        # Run the full setup pipeline: download → verify → extract.
        self.download()
        self.verify()
        self.extract()

if __name__ == "__main__":
    CLEVRDownloader("data/clevr").setup()
