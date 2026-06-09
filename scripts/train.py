"""
scripts/train.py — End-to-end training entry point.
"""

import os
import sys
import random
import time
import warnings

warnings.filterwarnings("ignore")

# Allow `from src.x import y` when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import CFG
from src.vocabulary import Vocabulary
from src.dataset import OCRDataset, collate_fn, get_train_transforms, get_val_transforms
from src.model import OCRModel
from src.train import train_one_epoch, validate, EarlyStopping


# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#  Main
def main() -> None:
    set_seed(CFG.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading dataset...")
    train_df = pd.read_csv(CFG.TRAIN_CSV_PATH)

    # Normalise column names — the raw CSV may have an unnamed index column
    if "Unnamed: 0" in train_df.columns:
        train_df = train_df.drop(columns=["Unnamed: 0"])
    train_df.columns = ["image", "label"]
    train_df["label"] = train_df["label"].astype(str)

    print(f"  Total samples : {len(train_df):,}")
    print(f"  Unique labels : {train_df['label'].nunique():,}")

    vocab = Vocabulary(train_df["label"].tolist())
    print(f"  Vocabulary    : {vocab.vocab_size} classes "
          f"({vocab.vocab_size - 1} chars + 1 blank)")

    train_data, val_data = train_test_split(
        train_df,
        test_size=CFG.VAL_SPLIT,
        random_state=CFG.SEED,
        shuffle=True,
    )
    train_data = train_data.reset_index(drop=True)
    val_data   = val_data.reset_index(drop=True)
    print(f"  Train / Val   : {len(train_data):,} / {len(val_data):,}")

    train_loader = DataLoader(
        OCRDataset(train_data, CFG.TRAIN_IMG_DIR, vocab,
                   transform=get_train_transforms()),
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        OCRDataset(val_data, CFG.TRAIN_IMG_DIR, vocab,
                   transform=get_val_transforms()),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    model = OCRModel(vocab_size=vocab.vocab_size).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(),
                      lr=CFG.LEARNING_RATE,
                      weight_decay=CFG.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=CFG.EPOCHS,
                                  eta_min=CFG.LR_MIN)
    scaler    = torch.cuda.amp.GradScaler()

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CFG.CHECKPOINT_DIR, "best_ocr_model.pt")
    early_stop      = EarlyStopping(patience=10, checkpoint_path=checkpoint_path)

    # Training loop
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}"
          f"  {'CER':>8}  {'Acc':>8}  {'LR':>10}  {'Time':>6}")
    print("─" * 68)

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()

        train_loss               = train_one_epoch(model, train_loader, optimizer,
                                                   scaler, device)
        val_loss, val_cer, val_acc, examples = validate(model, val_loader, vocab, device)
        current_lr               = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  {epoch:03d}  {train_loss:10.4f}  {val_loss:10.4f}"
              f"  {val_cer:8.4f}  {val_acc:8.4f}  {current_lr:10.2e}  {elapsed:5.1f}s")

        # Print sample predictions for the first 3 epochs and every 5th after that
        if epoch <= 3 or epoch % 5 == 0:
            print("  Sample predictions:")
            for pred, truth in examples[:4]:
                mark = "✓" if pred == truth else "✗"
                print(f"    {mark}  pred={pred!r:12s}  truth={truth!r}")
            print()

        if early_stop.step(val_cer, model):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nTraining complete.")
    print(f"Best CER        : {early_stop.best_score:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
