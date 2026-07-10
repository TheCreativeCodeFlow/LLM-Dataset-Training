# LLM Dataset Training Pipeline

## Project Structure

```
LLM-Dataset-Training/
├── data/
│   ├── raw/           # Raw downloaded datasets (gitignored)
│   ├── cleaned/       # Cleaned/deduplicated datasets (gitignored)
│   └── transformed/   # Transformed/formatted datasets (gitignored)
├── scripts/
│   ├── download.py    # Download raw datasets
│   ├── clean.py       # Deduplicate, filter, clean
│   ├── transform.py   # Format for training (chat format, etc.)
│   ├── train.py       # LoRA/QLoRA training script
│   ├── evaluate.py    # Evaluation script
└── serve.py           # FastAPI inference server
├── models/
│   ├── base/          # Base model weights (gitignored)
│   └── adapters/      # LoRA/QLoRA adapters (gitignored)
├── configs/
│   ├── train_config.yaml      # Training hyperparameters
│   └── dataset_config.yaml    # Dataset config & cleaning rules
├── configs/
│   ├── train_config.yaml
│   └── dataset_config.yaml
├── logs/              # Training logs (gitignored)
├── tests/             # Unit tests
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Download dataset
python scripts/download.py --config configs/dataset_config.yaml

# Clean dataset
python scripts/clean.py --config configs/dataset_config.yaml

# Transform for training
python scripts/transform.py --config configs/dataset_config.yaml

# Train LoRA adapter
python scripts/train.py --config configs/train_config.yaml

# Evaluate
python scripts/evaluate.py --config configs/train_config.yaml

# Serve API
python scripts/serve.py --config configs/train_config.yaml
```

## Configuration

- `configs/train_config.yaml` - Training hyperparameters, LoRA config, optimizer settings
- `configs/dataset_config.yaml` - Dataset source, cleaning rules, transformation format

## Environment

Copy `.env.example` to `.env` and add secrets (HF_TOKEN, etc.). Never commit secrets.

## Directories

- `data/*` - Dataset artifacts (gitignored for size/privacy)
- `models/base/*` - Base model weights (gitignored)
- `models/adapters/*` - LoRA adapters (gitignored)
- `logs/` - Training logs (gitignored)