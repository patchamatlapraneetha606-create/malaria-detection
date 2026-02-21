"""
Download malaria cell images from NIH (cell_images.zip).
Extracts to data/cell_images/Parasitized/ and Uninfected/ for training.
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from config import DATA_DIR, RAW_DATA_DIR, CLASS_NAMES

# NIH malaria cell images (same as TensorFlow Datasets source)
CELL_IMAGES_ZIP = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"


def download_malaria_sample(max_per_class: int = None):
    """
    Download NIH cell_images.zip and extract to RAW_DATA_DIR.
    If max_per_class is set, only copy that many images per class (saves disk).
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in CLASS_NAMES:
        (RAW_DATA_DIR / name).mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "cell_images.zip"
    extract_to = DATA_DIR / "cell_images_extract"

    if not zip_path.exists() or zip_path.stat().st_size < 100_000:
        print("Downloading malaria cell images from NIH (~337 MB). This may take a few minutes...")
        try:
            urllib.request.urlretrieve(CELL_IMAGES_ZIP, zip_path)
        except Exception as e:
            print(f"Download failed: {e}")
            print("You can manually download from:", CELL_IMAGES_ZIP)
            return None
        print("Download complete.")
    else:
        print("Using existing cell_images.zip")

    print("Extracting...")
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    # Zip contains "cell_images/Parasitized" and "cell_images/Uninfected"
    inner = extract_to / "cell_images"
    if not inner.exists():
        inner = extract_to
    for name in CLASS_NAMES:
        src = inner / name
        dst = RAW_DATA_DIR / name
        if not src.exists():
            continue
        # Clear destination so --max-per-class yields exactly that many
        for old in dst.glob("*"):
            old.unlink(missing_ok=True)
        files = list(src.glob("*.png")) + list(src.glob("*.jpg")) + list(src.glob("*.jpeg"))
        if max_per_class is not None:
            files = files[:max_per_class]
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"  {name}: {len(files)} images")

    if extract_to.exists():
        shutil.rmtree(extract_to)
    print(f"Done. Images are in {RAW_DATA_DIR}")
    return {name: len(list((RAW_DATA_DIR / name).glob("*.png")) + list((RAW_DATA_DIR / name).glob("*.jpg"))) for name in CLASS_NAMES}


def main():
    parser = argparse.ArgumentParser(description="Download malaria cell images for training")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max images per class (default: all)")
    args = parser.parse_args()
    download_malaria_sample(max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
