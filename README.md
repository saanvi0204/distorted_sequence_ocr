# Distorted OCR

> A deep learning OCR system for recognizing heavily distorted alphanumeric text sequences from noisy grayscale images.

This is an end-to-end optical character recognition pipeline designed to recover text sequences from grayscale images containing severe visual distortions.

The model is trained to handle challenging conditions such as:

- Background noise
- Character overlap
- Blur and visual artifacts
- Shape deformation
- Partial occlusion
- Irregular spacing and alignment

The system combines a custom CNN feature extractor, BiLSTM sequence modeling, Multi-Head Self-Attention, and CTC decoding to accurately reconstruct text from degraded visual inputs.

---

## Architecture

```
Input Image (1 × 100 × 200)
       │
       ▼
 CNN Backbone          — 5-block custom CNN with height-only pooling
       │                  preserves horizontal resolution → ~50 timesteps
       ▼
 AdaptiveAvgPool       — collapse height to 1; reshape to sequence (B, T, 512)
       │
       ▼
 BiLSTM (2 layers)     — sequential context in both directions
       │
       ▼
 Multi-Head Attention  — attend over all timesteps; residual add to LSTM output
       │
       ▼
 Linear + Log-Softmax  — character logits at each timestep
       │
       ▼
 CTC Decoder           — greedy decoding → predicted text
```

### Why a Custom CNN?

Many standard image classification backbones aggressively downsample image width, reducing the number of available timesteps for CTC decoding.

This project uses height-only pooling:

```python
MaxPool2d((2, 1))
```

which preserves horizontal information and produces approximately 50 decoding timesteps, enabling accurate recognition of long and repeated character sequences.

---

## Results

| Metric | Value |
|----------|----------|
| Character Error Rate (CER) | **0.0001** |
| Sequence Accuracy | **99.95%** |

---

## Dataset

Dataset:

https://drive.google.com/drive/folders/1lRUA-1uCCXfks8kpypFV-4f0UepWoLkU

Expected structure:

```text
data/
├── train_images/
│   ├── train-0.png
│   └── ...
├── test_images/
│   ├── test-0.png
│   └── ...
└── train-labels.csv
```

---

## Repository Structure

```text
distorted-ocr/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   ├── ocr_model.ipynb            # Full documented reproducible notebook
│
├── scripts/
│   ├── train.py                   # End-to-end training entry point
│   └── predict.py                 # Batch inference script
│
├── src/
│   ├── config.py                  # Hyperparameters and configuration
│   ├── dataset.py                 # Dataset loading and preprocessing
│   ├── model.py                   # CNN + BiLSTM + Attention OCR architecture
│   ├── train.py                   # Training and validation logic
│   ├── inference.py               # Decoding and evaluation utilities
│   └── vocabulary.py              # Character vocabulary and CTC tokeniser
│
├── checkpoints/
│   └── best_ocr_model.pt
│
└── data/
    ├── train_images/
    ├── test_images/
    └── train-labels.csv
```

---

# How to Run

## Option 1: Run the Complete Notebook

The easiest way to reproduce the project is to run:

```text
notebook/ocr-model-improved.ipynb
```

### Running on Kaggle

1. Create a new Kaggle Notebook.
2. Upload `ocr_model.ipynb`.
3. Attach the dataset.
4. Enable GPU:
   - Settings → Accelerator → GPU T4 x2
5. Run all cells.

The notebook contain the complete project workflow. It will:

- Load and preprocess the data
- Create the vocabulary
- Train the OCR model
- Evaluate validation performance
- Save the best checkpoint
- Generate predictions
- Perform error analysis

---

## Option 2: Run Using the Modular Python Code

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Dataset

Place the dataset inside:

```text
data/
├── train_images/
├── test_images/
└── train-labels.csv
```

### Train the Model

```bash
python scripts/train.py
```

### Generate Predictions

```bash
python scripts/predict.py
```

Predictions will be exported as:

```text
submission.csv
```

---

## Using the Pretrained Checkpoint

If `checkpoints/best_ocr_model.pt` is available:

```bash
python scripts/predict.py
```

This allows inference without retraining the model.

---

## Training Configuration

| Parameter | Value |
|----------|----------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Scheduler | Cosine Annealing |
| Loss Function | CTC Loss |
| Batch Size | 64 |
| Mixed Precision | FP16 |
| Gradient Clipping | 5.0 |
| Early Stopping | Patience = 15 |

---

## Environment

The project was developed and tested on Kaggle using Python 3.11 and PyTorch 2.x.

---

## References

This project was inspired by and builds upon ideas from the following foundational works in scene text recognition and sequence modeling:

1. Baoguang Shi, Xiang Bai, Cong Yao.
   **"An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition"** (2015)
   https://arxiv.org/abs/1507.05717

2. Baoguang Shi, Xinggang Wang, Pengyuan Lyu, Cong Yao, Xiang Bai.
   **"Attention-based Extraction of Structured Information from Street View Imagery"** (2017)
   https://arxiv.org/abs/1704.03549

3. Baoguang Shi, Mingkun Yang, Xinggang Wang, Pengyuan Lyu, Cong Yao, Xiang Bai.
   **"Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition"** (2018)
   https://arxiv.org/abs/1811.00751

These works provided valuable insights into sequence-based text recognition, feature extraction, attention mechanisms, and CTC-based decoding strategies that influenced the design and experimentation of this project.
