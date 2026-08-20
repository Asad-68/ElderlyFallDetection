"""
download_data.py — Download the Le2i Fall Detection dataset from Kaggle.

Usage:
    python src/download_data.py [--force]

Prerequisites:
    1. Install the kaggle package:  pip install kaggle
    2. Place your Kaggle API token at:
         - Linux / macOS : ~/.kaggle/kaggle.json
         - Windows       : C:\\Users\\<YourUser>\\.kaggle\\kaggle.json
       (Get the token from https://www.kaggle.com/settings → "Create New Token")
    3. Ensure the file permissions are restrictive:
         chmod 600 ~/.kaggle/kaggle.json   (Linux/macOS)

The script downloads and unzips the dataset into data/.
If data/ already contains files, it skips the download unless --force is passed.
"""

import argparse
import sys
from pathlib import Path

# Resolve project root so the script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

KAGGLE_DATASET = "tuyenldvn/falldataset-imvia"


def main():
    parser = argparse.ArgumentParser(
        description="Download the Le2i Fall Detection dataset from Kaggle."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data/ already contains files.",
    )
    args = parser.parse_args()

    # Check if data already exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DATA_DIR.iterdir())
    if existing and not args.force:
        print(f"[INFO] data/ already contains {len(existing)} item(s). "
              "Skipping download. Use --force to re-download.")
        return

    # Import kaggle only when needed so missing-token errors are obvious
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] The 'kaggle' package is not installed. Run:")
        print("        pip install kaggle")
        sys.exit(1)

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        print("Make sure your kaggle.json token is placed at the correct path:")
        print("  Linux/macOS : ~/.kaggle/kaggle.json")
        print("  Windows     : C:\\Users\\<YourUser>\\.kaggle\\kaggle.json")
        sys.exit(1)

    print(f"[INFO] Downloading dataset '{KAGGLE_DATASET}' into {DATA_DIR} ...")
    print("[INFO] This may take several minutes depending on your connection.")

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DATA_DIR),
        unzip=True,
        quiet=False,
    )

    print(f"[INFO] Download complete. Contents of data/:")
    for item in sorted(DATA_DIR.iterdir()):
        tag = "DIR " if item.is_dir() else "FILE"
        print(f"  [{tag}] {item.name}")

    print("[DONE] Dataset ready.")


if __name__ == "__main__":
    main()
