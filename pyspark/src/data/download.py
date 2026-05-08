import pandas as pd
from pathlib import Path
import sys
from urllib.request import urlretrieve
import zipfile
import requests

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
root = Path(__file__).resolve().parent.parent
dest_dir = root / "data/raw"
zip_dir = dest_dir / "online_retail_II.zip"
extract_dir = dest_dir / "extracted"

def zip_file_download(url = UCI_URL, dest = dest_dir, zip = zip_dir):
    dest.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        download = 0
        with open(zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    download += len(chunk)
                    pct = min(100, download * 100 / total) if total > 0 else 0
                    sys.stdout.write(f"\r  {pct:.1f}%")
                    sys.stdout.flush()

def extract(zip = zip_dir, file = extract_dir):
    file.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip) as z:
        z.extractall(file)