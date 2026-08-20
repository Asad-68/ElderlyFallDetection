# Elderly Fall Detection — CNN + Bi-LSTM Pipeline

> **CPU-only, proof-of-concept** fall detection from short video clips using
> MobileNetV2 feature extraction + Bidirectional LSTM classification.

Uses the **Le2i Fall Detection** dataset (Kaggle: `tuyenldvn/falldataset-imvia`).

---

## Project Structure

```
ElderlyFallDetection/
├── data/                  ← Downloaded dataset (gitignored)
├── features/              ← Cached CNN features (gitignored)
├── models/                ← Saved trained models
├── results/               ← Evaluation outputs (confusion matrix, etc.)
├── src/
│   ├── download_data.py   ← Kaggle API download script
│   ├── extract_features.py← Frame reading + MobileNetV2 feature extraction
│   ├── train.py           ← Bi-LSTM training
│   ├── evaluate.py        ← Evaluation, metrics, confusion matrix
│   └── utils.py           ← Shared helpers (video discovery, frame reading)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Start

### 1. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs **`tensorflow-cpu`** — no CUDA/cuDNN required.

### 3. Set Up Kaggle API Token

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New Token**
2. Download `kaggle.json`
3. Place it at:
   - **Windows:** `C:\Users\<YourUser>\.kaggle\kaggle.json`
   - **macOS / Linux:** `~/.kaggle/kaggle.json`
4. (Linux/macOS) Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 4. Download the Dataset

```bash
python src/download_data.py
```

Re-run with `--force` to re-download.

### 5. Extract Features

```bash
python src/extract_features.py
```

This reads video clips, runs MobileNetV2 inference per frame, and saves
features to `features/features.npz`. On CPU, expect **~10-20 min** for the
default 120 clips.

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--max-clips` | 120 | Total clips to process |
| `--num-frames` | 16 | Frames sampled per clip |
| `--batch-size` | 32 | CNN batch size |
| `--force` | — | Re-extract even if cache exists |

### 6. Train the Model

```bash
python src/train.py
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Maximum training epochs |
| `--batch-size` | 16 | Training batch size |
| `--test-size` | 0.2 | Test set fraction |
| `--patience` | 8 | Early stopping patience |

### 7. Evaluate

```bash
python src/evaluate.py
```

Prints a classification report and saves `results/confusion_matrix.png`.

---

## Architecture

```
Video clip (16 frames, 160×160)
    │
    ▼
MobileNetV2 (frozen, ImageNet)     ← per-frame feature extraction
    │  output: (16, 1280)
    ▼
Bidirectional LSTM (64 units)       ← temporal modeling
    │  output: (128,)
    ▼
Dropout (0.4) → Dense (32, ReLU) → Dropout (0.3)
    │
    ▼
Dense (1, Sigmoid)                  ← binary: fall / not_fall
```

---

## Labelling Strategy

1. **Annotation files** (priority): Each scene folder may contain
   `Annotation_files/*.txt` with fall start/end frames → label = **fall**.
2. **Filename heuristic** (fallback): If the video path contains `"fall"`
   (case-insensitive) → **fall**; otherwise → **not_fall**.

This is documented in detail in `src/utils.py`.

---

## CPU-Only Notes

- `tensorflow-cpu` is installed — no GPU drivers needed.
- Feature extraction is the bottleneck (~10-20 min for 120 clips).
  Bi-LSTM training is fast (~1-2 min).
- Features are cached to disk so training can be re-run instantly.

---

## Scaling Up

To move beyond proof-of-concept:

| What | How |
|------|-----|
| More data | Increase `--max-clips` (or remove the cap) |
| GPU acceleration | Switch to `tensorflow` (with CUDA), use larger batches |
| Better accuracy | Unfreeze top MobileNetV2 layers, use more frames, add data augmentation |
| Larger model | Stack 2 Bi-LSTM layers, increase units to 128+ |
| Cloud training | Use Google Colab (free GPU) or a cloud VM |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned versions
- No GPU required
