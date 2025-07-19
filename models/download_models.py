#!/usr/bin/env python3
"""
Download pre-trained model files.

This script downloads the required .pth model files from Google Drive
and places them in the current directory (models/).
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
    """Download all required model files."""
    # Use current directory (models/)
    models_dir = Path(".")
    
    # Model file configurations
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
    
    print("PerToon Model Downloader")
    print("=" * 40)
    
    for model in models:
        file_path = models_dir / model["name"]
        
        # Check if file already exists
        if file_path.exists():
            print(f"✓ {model['name']} already exists, skipping...")
            continue
        
        print(f"Downloading {model['description']}...")
        print(f"   File: {model['name']}")
        
        try:
            success = download_file_from_google_drive(model["file_id"], file_path)
            
            if success and file_path.exists():
                file_size = file_path.stat().st_size
                print(f"✓ Downloaded successfully ({file_size:,} bytes)")
            else:
                print(f"Failed to download {model['name']}")
                return False
                
        except Exception as e:
            print(f"Error downloading {model['name']}: {str(e)}")
            return False
    
    print("\nAll model files downloaded successfully!")
    print(f"Models saved in: {models_dir.absolute()}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 