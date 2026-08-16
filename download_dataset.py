"""
Downloads the NIH ChestX-ray14 dataset via kagglehub and prints the path
you should set as CXR_DATA_DIR (or paste into config.py's DATA_DIR).

Requires a Kaggle account + API credentials set up once:
  1. Go to https://www.kaggle.com/settings -> "Create New Token"
  2. This downloads kaggle.json -- place it at:
       Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
       macOS/Linux: ~/.kaggle/kaggle.json
  (kagglehub will prompt you interactively the first time if it's missing)

Usage:
    pip install kagglehub
    python download_dataset.py
"""

import os

import kagglehub


def main():
    print("Downloading NIH ChestX-ray14 dataset via kagglehub ...")
    print("(This is ~45GB and may take a while the first time; kagglehub")
    print(" caches it, so subsequent runs are instant.)\n")

    path = kagglehub.dataset_download("nih-chest-xrays/data")
    print(f"\nDataset downloaded to: {path}")

    # Sanity check for the files our pipeline expects
    expected = ["Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"]
    missing = [f for f in expected if not os.path.exists(os.path.join(path, f))]
    if missing:
        print(f"\nWARNING: expected files not found at top level: {missing}")
        print("Listing top-level contents so you can locate them:")
        for entry in sorted(os.listdir(path))[:30]:
            print(f"  {entry}")
    else:
        print("\nAll expected metadata files found.")

    print("\nNext step -- set this as your dataset directory, e.g.:")
    print(f'  export CXR_DATA_DIR="{path}"      # macOS/Linux')
    print(f'  $env:CXR_DATA_DIR = "{path}"       # Windows PowerShell')
    print("\nOr just paste this path into DATA_DIR in config.py directly.")


if __name__ == '__main__':
    main()
