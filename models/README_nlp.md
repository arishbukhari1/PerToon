# NLP Models for PerToon (GoEmotions GPT-2)

This directory contains the minimal files required for the GoEmotions-based GPT-2 models used in PerToon for emotion-driven captioning.

## Model Subdirectories

- `gpt2-base-goemotions/`
- `gpt2-medium-goemotions/`

Each contains the minimal set of files needed for inference:
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `vocab.json`
- `model.safetensors` (downloaded via script or manually)

## Download Instructions

The config and tokenizer files are already committed. To download the model weights (`model.safetensors`):

### Option 1: Automatic Download (Recommended)

Run the download script from the `models/` directory:

```bash
cd models
python download_nlp_models.py
```

Or from the project root:

```bash
python models/download_nlp_models.py
```

This will automatically download the following files:
- `gpt2-base-goemotions/model.safetensors` ([Google Drive link](https://drive.google.com/file/d/17IpJtKG6ZHy_IxtkQ36FiIAiChg5hrK0/view?usp=sharing))
- `gpt2-medium-goemotions/model.safetensors` ([Google Drive link](https://drive.google.com/file/d/1CTMlPJ4MWSu7xxbX65hWb_TpD-U7GUIr/view?usp=sharing))

### Option 2: Manual Download

1. Download `model.safetensors` for each model from the links above.
2. Place them in the appropriate subdirectory (e.g., `models/gpt2-base-goemotions/`).

## What is committed vs. ignored?

- **Committed:** Minimal config/tokenizer files.
- **Downloaded:** `model.safetensors` for each model (not committed due to size).
- **Ignored:** All training checkpoints, optimizer/scheduler/trainer state files, and any large files not needed for inference (see `.gitignore`).

## Example Directory Structure

```
models/
├── README_nlp.md (this file)
├── download_nlp_models.py (NLP model download script)
├── gpt2-base-goemotions/
│   ├── config.json
│   ├── generation_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── model.safetensors
├── gpt2-medium-goemotions/
│   ├── config.json
│   ├── generation_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── model.safetensors
```

## Notes
- If you add new models, update this README and the downloader script.
- For training or resuming training, use Google Drive to store and share full checkpoints.
