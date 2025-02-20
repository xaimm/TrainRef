# TrainRef
This is the official repository for the paper "Training Beyond Label Misinformation: Reliable Reference is All You Need" (Under ICML25 review)


## Overview
This repository contains scripts for influence-based label noise detection and correction, followed by fine-tuning a model on the refined dataset. The two main scripts are:
- `train_script.py`: Performs label noise detection and correction using influence functions.
- `finetune_script.py`: Fine-tunes a model using the cleaned dataset obtained from the previous step.

## Dependencies
Ensure you have the required dependencies installed. You can install them using the provided `requirements.txt` file.

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install manually using:

```bash
pip install torch torchvision timm numpy pandas tqdm pillow transformers
```

## Usage

### 1. Reference-Based Label Noise Detection and Correction
Run the `train_script.py` to detect and correct label noise in the dataset.

```bash
python train_script.py --dataset DATASET_PATH --output CLEANED_DATASET_PATH --epochs 50 --batch_size 32 --model_type vit_small --patch_size 16 --checkpoint_key teacher --pretrained_weights PRETRAINED_PATH
```

**Arguments:**
- `--dataset`: Path to the dataset containing noisy labels.
- `--output`: Path to save the cleaned dataset after correction.
- `--epochs`: Number of training epochs (default: 50).
- `--batch_size`: Batch size for training (default: 32).
- `--model_type`: Type of model used for training (default: vit_small).
- `--patch_size`: Patch size of the vision transformer model (default: 16).
- `--checkpoint_key`: Checkpoint key to load the pretrained model (default: teacher).
- `--pretrained_weights`: Path to the pretrained model weights.

This script applies TrainRef to detect mislabeled samples and correct them.

### 2. Fine-Tuning on the Cleaned Dataset
After obtaining a corrected dataset, fine-tune a model using `finetune_script.py`.

```bash
python finetune_script.py --dataset CLEANED_DATASET_PATH --model MODEL_PATH --epochs 30 --batch_size 32 --learning_rate 0.001 --momentum 0.9 --scheduler_step 10 --gamma 0.1
```

**Arguments:**
- `--dataset`: Path to the cleaned dataset.
- `--model`: Path to a pre-trained model (if applicable).
- `--epochs`: Number of fine-tuning epochs (default: 30).
- `--batch_size`: Batch size for fine-tuning (default: 32).
- `--learning_rate`: Learning rate for optimizer (default: 0.001).
- `--momentum`: Momentum for SGD optimizer (default: 0.9).
- `--scheduler_step`: Step size for learning rate scheduler (default: 10).
- `--gamma`: Decay rate for learning rate scheduler (default: 0.1).



