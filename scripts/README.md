# Scripts Guide

This directory contains Python scripts for working with the signature verification system.

## Available Scripts

### 1. Data Preparation (`prepare_data.py`)

Prepares training and test datasets from raw signature images.

```bash
python scripts/prepare_data.py
```

This script:
- Generates positive pairs (same person signatures)
- Generates negative pairs (different person signatures)
- Splits data into train/test sets (95%/5% by default)
- Saves to `./Data/train_index.pkl` and `./Data/test_index.pkl`

**Prerequisites**: Extract `Data_raw.7z` first!

### 2. Training (`train.py`)

Train the Siamese Network model.

```bash
# Basic usage
python scripts/train.py

# Custom configuration
python scripts/train.py \
    --epochs 10 \
    --batch-size 16 \
    --lr 0.0005 \
    --device cuda
```

**Arguments**:
- `--data`: Path to training data (default: `./Data/train_index.pkl`)
- `--model-dir`: Directory to save checkpoints (default: `./Models`)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use - `auto`, `cuda`, or `cpu` (default: auto)

**Output**: Model checkpoints saved to `./Models/checkpoint_epoch_N`

### 3. Evaluation (`evaluate.py`)

Evaluate a trained model and generate ROC/PR curves.

```bash
# Evaluate a model
python scripts/evaluate.py --model ./Models/checkpoint_epoch_4

# Custom configuration
python scripts/evaluate.py \
    --model ./Models/checkpoint_epoch_9 \
    --data ./Data/test_index.pkl \
    --output ./my_results.png \
    --batch-size 16
```

**Arguments**:
- `--model`: Path to model checkpoint (required)
- `--data`: Path to test data (default: `./Data/test_index.pkl`)
- `--output`: Output plot path (default: `./evaluation_results.png`)
- `--batch-size`: Batch size (default: 8)
- `--device`: Device to use (default: auto)

**Output**:
- ROC and Precision-Recall curves saved as PNG
- Evaluation metrics printed to console

## Complete Workflow

```bash
# 1. Prepare data
python scripts/prepare_data.py

# 2. Train model
python scripts/train.py --epochs 5 --batch-size 8

# 3. Evaluate model
python scripts/evaluate.py --model ./Models/checkpoint_epoch_4

# 4. View results
# Check evaluation_results.png
```

## Requirements

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Notes

- All scripts support CUDA acceleration if available
- Progress is printed during execution
- Model checkpoints are saved after each epoch
- The evaluation script generates high-resolution plots (300 DPI)
