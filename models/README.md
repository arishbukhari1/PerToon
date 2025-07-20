# Vision Models for Cartoonization

This directory contains the pre-trained model files required for the cartoonization system.

## Required Models

The following model files are needed:

- `genA2B_final.pth` - Generator model for converting from human image to cartoon
- `genB2A_final.pth` - Generator model for converting from cartoon to human image

## Download Instructions

These model files are too large to be included in the Git repository. To download them:

### Option 1: Automatic Download (Recommended)

Run the download script from this directory:

```bash
cd models
python download_vision_models.py
```

Or from the project root:

```bash
python models/download_vision_models.py
```

This will automatically download both model files from Google Drive and place them in this directory.

### Option 2: Manual Download

1. Download `genA2B_final.pth` from: https://drive.google.com/file/d/15bSGDiqQhLXh35eSUvJNdymakHrzeZ5c/view?usp=sharing
2. Download `genB2A_final.pth` from: https://drive.google.com/file/d/12NpKowJwEDIE1wqtPqCdPEMcLoJGzHeQ/view?usp=sharing
3. Place both files in this `models/` directory

## Verification

After downloading, you should have:

```
models/
├── README.md (this file)
├── download_vision_models.py (vision model download script)
├── genA2B_final.pth
└── genB2A_final.pth
```

The model files are typically several hundred MB each.

## Note

These files are excluded from Git tracking via `.gitignore` to prevent repository bloat. Each model file contains the trained weights for the cartoon generation neural network. 