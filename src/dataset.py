"""
dataset.py — Dataset, DataLoader collation, and image transforms.
"""

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from .vocabulary import Vocabulary


# Transforms
def get_train_transforms() -> A.Compose:
    """
    Augmentation pipeline applied to training images only.

    Each transform targets a specific distortion type present in the dataset:
      - GaussNoise          → noisy backgrounds
      - GaussianBlur / MotionBlur → blur and image artifacts
      - GridDistortion      → warped / deformed characters
      - ElasticTransform    → shape deformation
      - ShiftScaleRotate    → alignment and position variation
      - CoarseDropout       → occlusion and missing character regions
      - RandomBrightnessContrast / RandomGamma → contrast variation

    Validation and test images receive only normalization (see get_val_transforms).
    """
    return A.Compose([
        A.GaussNoise(std_range=(0.02, 0.08), p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(p=1.0),
        ], p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=3,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.02, 0.06),
            hole_width_range=(0.02, 0.08),
            fill=0,
            p=0.3,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Minimal pipeline for validation and test images — normalization only."""
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


# Dataset

class OCRDataset(Dataset):
    """
    Loads grayscale OCR images and encodes their text labels.

    Args:
        df:        DataFrame with columns ['image', 'label'] (label omitted for test).
        image_dir: Directory containing the image files.
        vocab:     Vocabulary instance used for label encoding.
        transform: Albumentations transform pipeline.
        is_test:   If True, labels are not loaded — returns (image, filename) pairs.
    """

    def __init__(self, df, image_dir: str, vocab: Vocabulary,
                 transform=None, is_test: bool = False) -> None:
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.vocab     = vocab
        self.transform = transform
        self.is_test   = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image"])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.is_test:
            return image, row["image"]

        label = torch.tensor(self.vocab.encode(row["label"]), dtype=torch.long)
        return image, label, row["label"]


# Collation
def collate_fn(batch):
    """
    Collate training / validation batches for CTC loss.

    CTC loss expects:
      - targets:        all label sequences concatenated into a single 1-D tensor
      - target_lengths: length of each individual label sequence
    """
    images, labels, raw_labels = zip(*batch)
    images         = torch.stack(images)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets        = torch.cat(labels)
    return images, targets, target_lengths, raw_labels


def collate_fn_test(batch):
    """Collate test batches — no labels, returns (images, filenames)."""
    images, filenames = zip(*batch)
    return torch.stack(images), list(filenames)
