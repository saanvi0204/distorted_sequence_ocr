"""
train.py — Training loop, validation loop, and early stopping.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from .inference import greedy_decode, compute_cer, sequence_accuracy


# Loss
_ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)


def compute_ctc_loss(log_probs: torch.Tensor,
                     targets: torch.Tensor,
                     target_lengths: torch.Tensor) -> torch.Tensor:
    T, B, _ = log_probs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=targets.device)
    return _ctc_loss(log_probs, targets, input_lengths, target_lengths)


# Early Stopping
class EarlyStopping:
    """
    Monitors validation CER and saves the best model checkpoint.

    Args:
        patience:        Number of epochs to wait without improvement.
        min_delta:       Minimum absolute improvement to count as progress.
        checkpoint_path: Path where the best model state dict is saved.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 checkpoint_path: str = "checkpoints/best_ocr_model.pt") -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = checkpoint_path
        self.best_score = None
        self.counter    = 0

    def step(self, val_cer: float, model: nn.Module) -> bool:
        """
        Call after each epoch.  Returns True when training should stop.
        Saves a checkpoint whenever a new best CER is reached.
        """
        if self.best_score is None or val_cer < self.best_score - self.min_delta:
            self.best_score = val_cer
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            print(f"  ✓ Checkpoint saved (CER={val_cer:.4f})")
        else:
            self.counter += 1
            print(f"  Early stop counter: {self.counter}/{self.patience}")

        return self.counter >= self.patience


# Train / Validate
def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    """
    Run one full pass over the training set.

    Returns:
        Average CTC loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, targets, target_lengths, _ in loader:
        images         = images.to(device)
        targets        = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            log_probs = model(images)
            loss      = compute_ctc_loss(log_probs, targets, target_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, vocab, device) -> tuple:
    """
    Evaluate the model on the validation set.

    Returns:
        (avg_loss, avg_cer, seq_accuracy, sample_predictions)
        sample_predictions is a list of (pred, truth) pairs for inspection.
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for images, targets, target_lengths, raw_labels in loader:
        images         = images.to(device)
        targets        = targets.to(device)
        target_lengths = target_lengths.to(device)

        log_probs   = model(images)
        loss        = compute_ctc_loss(log_probs, targets, target_lengths)
        total_loss += loss.item()

        preds = greedy_decode(log_probs, vocab)
        all_preds.extend(preds)
        all_targets.extend(raw_labels)

    avg_loss = total_loss / len(loader)
    avg_cer  = compute_cer(all_preds, all_targets)
    seq_acc  = sequence_accuracy(all_preds, all_targets)

    # Random sample of predictions for console logging
    idx      = np.random.choice(len(all_preds), min(10, len(all_preds)), replace=False)
    examples = [(all_preds[i], all_targets[i]) for i in idx]

    return avg_loss, avg_cer, seq_acc, examples
