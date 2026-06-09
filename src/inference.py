"""
inference.py — Greedy CTC decoding, batch prediction, and evaluation metrics.
"""

import numpy as np
import editdistance
import torch

from .vocabulary import Vocabulary


# Decoding
def greedy_decode(log_probs: torch.Tensor, vocab: Vocabulary) -> list[str]:
    """
    Greedy CTC decoding — fast and deterministic.

    Algorithm:
      1. At each timestep, select the character with the highest log-probability.
      2. Collapse consecutive duplicate predictions (CTC merge rule).
      3. Remove blank tokens (index 0).
      4. Convert indices to characters via the vocabulary.

    Args:
        log_probs: Model output of shape (T, B, vocab_size).
        vocab:     Vocabulary instance for index-to-character conversion.

    Returns:
        List of decoded strings, one per sample in the batch.
    """
    pred_indices = log_probs.argmax(dim=2).cpu().numpy()   # (T, B)
    predictions  = []

    for b in range(pred_indices.shape[1]):
        text = vocab.decode(pred_indices[:, b].tolist())
        predictions.append(text)

    return predictions


@torch.no_grad()
def predict(model, loader, vocab: Vocabulary, device) -> dict[str, str]:
    """
    Run inference over a test DataLoader and return a filename → prediction dict.

    Args:
        model:  Trained OCRModel in eval mode.
        loader: DataLoader created with collate_fn_test.
        vocab:  Vocabulary instance.
        device: torch.device.

    Returns:
        Dict mapping each image filename to its predicted text sequence.
    """
    model.eval()
    predictions: dict[str, str] = {}

    for images, filenames in loader:
        images    = images.to(device)
        log_probs = model(images)
        preds     = greedy_decode(log_probs, vocab)

        for fname, pred in zip(filenames, preds):
            predictions[fname] = pred

    return predictions


# Metrics
def compute_cer(predictions: list[str], targets: list[str]) -> float:
    """
    Character Error Rate (CER) = mean(Levenshtein distance / len(target)) over all samples.
    A CER of 0.0 means every character in every prediction is correct.
    Lower is better.

    Args:
        predictions: List of predicted text sequences.
        targets:     List of ground-truth text sequences.

    Returns:
        Mean CER across all samples (float in [0, ∞)).
    """
    scores = [
        editdistance.eval(pred, target) / len(target)
        for pred, target in zip(predictions, targets)
    ]
    return float(np.mean(scores))


def sequence_accuracy(predictions: list[str], targets: list[str]) -> float:
    """
    Fraction of predictions that are an exact match with the ground truth.

    Returns:
        Float in [0.0, 1.0].
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)
