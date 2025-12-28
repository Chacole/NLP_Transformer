# Vietnamese-English Neural Machine Translation with Transformer

A deep learning project implementing **Vietnamese-English Machine Translation** using the Transformer architecture from scratch. This project was developed and trained on **Kaggle** using GPU acceleration.

---

## ⚠️ Important: Kaggle Execution Environment

> **This project was developed and executed entirely on Kaggle.**
>
> All notebooks are designed to run on Kaggle's platform with GPU acceleration enabled. The training process utilizes Kaggle's free GPU resources (P100/T4), and all data paths are configured for the Kaggle environment.

### Running on Kaggle

1. Upload the notebooks to Kaggle
2. Add the required datasets (see each task's section below)
3. Enable GPU acceleration: **Settings → Accelerator → GPU**
4. Run all cells

---

## 📁 Project Structure

```
NLP_Transformer-1/
├── README.md
├── nlp-transformer.ipynb     # Task 1: Custom Transformer (from scratch)
└── vlsp-2025.ipynb           # Task 2: VLSP 2025 Medical Translation
```

---

# Task 1: Custom Transformer Implementation

**Notebook:** `nlp-transformer.ipynb`

## Overview

A **from-scratch implementation** of the Transformer architecture for Vietnamese-English translation. This notebook covers the complete pipeline from data preprocessing to model evaluation.

## What's Covered

1. **Data Loading & Exploration** - Load the VLSP parallel corpus
2. **Data Preprocessing** - Clean and filter sentence pairs
3. **Tokenization** - Build BPE (Byte-Pair Encoding) tokenizers
4. **Dataset Preparation** - Create PyTorch datasets and dataloaders
5. **Model Architecture** - Implement Transformer from scratch
6. **Training** - Train with label smoothing and learning rate scheduling
7. **Evaluation** - Calculate BLEU scores
8. **Inference** - Translate new sentences with beam search

## Dataset

Uses the **VLSP Vietnamese-English Parallel Corpus**:

- **Kaggle Path:** `/kaggle/input/vie-eng/`

| Split    | Raw Size      | After Cleaning |
| -------- | ------------- | -------------- |
| Training | 500,000 pairs | ~343,110 pairs |
| Test     | 3,000 pairs   | ~2,976 pairs   |

**Cleaning Pipeline:**

- Unicode normalization (NFC form)
- Smart quote replacement
- Control character removal
- Punctuation filtering
- Sentence length filtering (3-100 words)
- Duplicate removal

## Model Configuration

| Parameter                 | Value        |
| ------------------------- | ------------ |
| Model Dimension (d_model) | 512          |
| Number of Layers          | 6            |
| Attention Heads           | 8            |
| Feed-Forward Dimension    | 2048         |
| Dropout                   | 0.1          |
| Max Sequence Length       | 128          |
| Vocabulary Size           | 32,000 (BPE) |
| Batch Size                | 64           |

## Training on Kaggle

- **GPU:** NVIDIA Tesla P100 / T4
- **Training Time:** ~2 hours per epoch
- **Optimizer:** Adam (β1=0.9, β2=0.98, ε=1e-9)
- **Label Smoothing:** 0.1

## Features

- ✅ Complete Transformer architecture implementation
- ✅ Multi-Head Self-Attention & Cross-Attention
- ✅ Positional Encoding (Sinusoidal)
- ✅ BPE Tokenization
- ✅ Beam Search Decoding
- ✅ BLEU Score Evaluation
- ✅ Training visualizations

---

# Task 2: VLSP 2025 Medical Translation

**Notebook:** `vlsp-2025.ipynb`

## Overview

An **optimized implementation** for the **VLSP 2025 Shared Task** on Medical Domain Translation (English ↔ Vietnamese). This notebook is designed for extended training with checkpoint resume capability.

## Key Features

- **Checkpoint Resume:** Continue training from saved checkpoints
- **Best Model Saving:** Automatically saves the model with lowest validation loss
- **Optimized Training Loop:** Efficient training with progress tracking
- **Validation Monitoring:** Track validation loss to prevent overfitting

## Dataset

Uses the VLSP 2025 train-and-test dataset:

- **Kaggle Path:** `/kaggle/input/train-and-test/`

| File                 | Description                   |
| -------------------- | ----------------------------- |
| `train.en.txt`       | English training sentences    |
| `train.vi.txt`       | Vietnamese training sentences |
| `public_test.en.txt` | English test sentences        |
| `public_test.vi.txt` | Vietnamese test sentences     |

## Configuration

```python
class Config:
    # Paths (Kaggle)
    TRAIN_EN = '/kaggle/input/train-and-test/train.en.txt'
    TRAIN_VI = '/kaggle/input/train-and-test/train.vi.txt'

    # Checkpoint Settings
    RESUME = False  # Set to "False" during the first training session, Set to "True" to continue from checkpoint
    RESUME_PATH = '' # example:'/kaggle/input/checkpoint/vlsp_checkpoint_last.pth'
    SAVE_PATH = '/kaggle/working/vlsp_checkpoint_last.pth'
    BEST_MODEL_PATH = '/kaggle/working/vlsp_best_model.pth'

    # Model Parameters
    SRC_VOCAB_SIZE = 32000
    TGT_VOCAB_SIZE = 32000
    D_MODEL = 512
    N_LAYERS = 6
    HEADS = 8
    FF_DIM = 2048
    DROPOUT = 0.1
    MAX_LEN = 128

    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 5
    LABEL_SMOOTHING = 0.1
    LR = 0.0001
```

## Checkpoint Resume Workflow

1. **First Run:** Train for N epochs, saves checkpoint
2. **Download:** Save `vlsp_checkpoint_last.pth` as a Kaggle dataset
3. **Next Run:** Set `RESUME = True` and update `RESUME_PATH`
4. **Continue:** Training resumes from last epoch

## Features

- ✅ Checkpoint save/load functionality
- ✅ Best model tracking
- ✅ Gradient clipping for stability
- ✅ 90/10 train/validation split
- ✅ Beam search evaluation
- ✅ BLEU score calculation with sacrebleu

---

## 🏗️ Shared Architecture Details

Both tasks use the same Transformer architecture:

### Components

- **Positional Encoding:** Sinusoidal position embeddings
- **Multi-Head Attention:** 8 parallel attention heads
- **Layer Normalization:** Pre-norm architecture
- **Feed-Forward Networks:** Two-layer MLP with ReLU

### Tokenization

- **Byte-Pair Encoding (BPE)** with 32,000 vocabulary
- Special tokens: `[PAD]`, `[SOS]`, `[EOS]`, `[UNK]`

---

## � Quick Start

### Prerequisites

```bash
pip install torch tokenizers sacrebleu nltk tqdm matplotlib numpy
```

### Running on Kaggle (Recommended)

1. **Fork the notebook** on Kaggle
2. **Add the appropriate dataset** for your task
3. **Enable GPU:** Settings → Accelerator → GPU P100
4. **Run all cells**

---

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [VLSP 2025 Shared Task](https://vlsp.org.vn/) - Vietnamese Language and Speech Processing

---

## 👤 Author

Developed for the NLP 2025 Final Project

---

## 🙏 Acknowledgments

- **Kaggle** for providing free GPU resources for training
- **VLSP** for the Vietnamese-English parallel corpus
- **HuggingFace** for the Tokenizers library
