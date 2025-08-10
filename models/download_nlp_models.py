#!/usr/bin/env python3
"""
Download pre-trained NLP model weights for PerToon.

This script downloads the required model.safetensors files for the gpt2-base-goemotions and gpt2-medium-goemotions models from Google Drive and places them in the appropriate subdirectories under models/.
"""

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("❌ Error: 'gdown' library not found.")
    print("Please install it with: pip install gdown")
    sys.exit(1)

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)
    return True

def main():
    """Download all required NLP model weights."""
    models = [
        {
            "dir": "gpt2-base-goemotions",
            "files": [
                {"name": "model.safetensors", "file_id": "17IpJtKG6ZHy_IxtkQ36FiIAiChg5hrK0"}
            ]
        },
        {
            "dir": "gpt2-medium-goemotions",
            "files": [
                {"name": "model.safetensors", "file_id": "1CTMlPJ4MWSu7xxbX65hWb_TpD-U7GUIr"}
            ]
        }
    ]

    print("PerToon NLP Model Downloader")
    print("=" * 40)

    for model in models:
        model_dir = Path(model["dir"])
        model_dir.mkdir(exist_ok=True)
        for file in model["files"]:
            file_path = model_dir / file["name"]
            if file_path.exists():
                print(f"{file_path} already exists - skipping")
                continue
            print(f"📥 Downloading {file_path}")
            try:
                download_file_from_google_drive(file["file_id"], file_path)
                print(f"Successfully downloaded {file_path}")
            except Exception as e:
                print(f"Failed to download {file_path}: {e}")
                return False
    print("\nAll NLP model downloads completed!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
