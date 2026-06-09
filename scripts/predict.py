"""
scripts/predict.py — Load a trained checkpoint and generate submission.csv.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.vocabulary import Vocabulary
from src.dataset import OCRDataset, collate_fn_test, get_val_transforms
from src.model import OCRModel
from src.inference import predict


# CLI 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OCR predictions for the test set."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(CFG.CHECKPOINT_DIR, "best_ocr_model.pt"),
        help="Path to the saved model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(CFG.SUBMISSION_DIR, "submission.csv"),
        help="Destination path for the submission CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CFG.BATCH_SIZE,
        help="Inference batch size (default: %(default)s).",
    )
    return parser.parse_args()


# Main 
def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output     : {args.output}")

    # Rebuild vocabulary from training labels
    print("\nBuilding vocabulary from training labels...")
    train_df = pd.read_csv(CFG.TRAIN_CSV_PATH)
    if "Unnamed: 0" in train_df.columns:
        train_df = train_df.drop(columns=["Unnamed: 0"])
    train_df.columns = ["image", "label"]
    train_df["label"] = train_df["label"].astype(str)

    vocab = Vocabulary(train_df["label"].tolist())
    print(f"Vocabulary size: {vocab.vocab_size} "
          f"({vocab.vocab_size - 1} chars + 1 blank)")

    # Test dataset
    test_files = sorted(os.listdir(CFG.TEST_IMG_DIR))
    test_df    = pd.DataFrame({"image": test_files})
    print(f"\nTest images: {len(test_df):,}")

    test_loader = DataLoader(
        OCRDataset(test_df, CFG.TEST_IMG_DIR, vocab,
                   transform=get_val_transforms(), is_test=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_test,
    )

    # Load model
    model = OCRModel(vocab_size=vocab.vocab_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded  ✓")

    # Inference
    print("\nRunning inference...")
    predictions = predict(model, test_loader, vocab, device)
    print(f"Predictions generated: {len(predictions):,}")

    # Preview the first five predictions
    print("\nFirst 5 predictions:")
    for fname in test_files[:5]:
        print(f"  {fname}  →  {predictions[fname]}")

    # Build and validate submission CSV
    submission = pd.DataFrame({
        "image":      test_files,
        "prediction": [predictions[f] for f in test_files],
    })

    assert list(submission.columns) == ["image", "prediction"], \
        "Column names do not match the required format."
    assert submission["image"].is_unique, \
        "Duplicate image filenames found in submission."
    assert submission["prediction"].notna().all(), \
        "Some predictions are missing (NaN)."

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved → {args.output}")
    print(f"Rows             : {len(submission):,}")

    # Prediction length distribution — a quick sanity check
    length_counts = submission["prediction"].str.len().value_counts().sort_index()
    print("\nPrediction length distribution:")
    for length, count in length_counts.items():
        print(f"  len={length}  →  {count:,} images")


if __name__ == "__main__":
    main()
