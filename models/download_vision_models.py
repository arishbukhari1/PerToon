#!/usr/bin/env python3
"""
Download pre-trained vision model files.

This script downloads the required .pth vision model files from Google Drive
for cartoonization and places them in the current directory (models/).
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
    """Download a file from Google Drive using gdown - same pattern as other projects."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)
    return True

def main():
    """Download all required vision model files."""
    # Use current directory (models/)
    models_dir = Path(".")
    
    # Vision model file configurations
    models = [
        {
            "name": "genA2B_final.pth",
            "file_id": "15bSGDiqQhLXh35eSUvJNdymakHrzeZ5c",
            "description": "Generator human image to cartoon model"
        },
        {
            "name": "genB2A_final.pth", 
            "file_id": "12NpKowJwEDIE1wqtPqCdPEMcLoJGzHeQ",
            "description": "Generator cartoon to human image model"
        }
    ]
    
    print("PerToon Vision Model Downloader")
    print("=" * 40)
    
    for model in models:
        file_path = models_dir / model["name"]
        
        # Check if file already exists
        if file_path.exists():
            print(f"{model['name']} already exists - skipping")
            continue
            
        print(f"📥 Downloading {model['name']} - {model['description']}")
        try:
            download_file_from_google_drive(model["file_id"], file_path)
            print(f"Successfully downloaded {model['name']}")
        except Exception as e:
            print(f"Failed to download {model['name']}: {e}")
            return False
    
    print("\nAll vision model downloads completed!")
    print("\nDownloaded models:")
    for model in models:
        file_path = models_dir / model["name"]
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  • {model['name']} ({size_mb:.1f} MB)")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 