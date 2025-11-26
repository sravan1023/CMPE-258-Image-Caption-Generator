"""
Dataset and dataloader helpers for the baseline-only model.
"""

from __future__ import annotations

import os
import pickle
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageCaptionDataset(Dataset):
    """Wrap images and processed caption indices."""

    def __init__(
        self,
        image_dir: str,
        processed_captions: dict,
        vocab_data: dict,
        image_names,
        max_caption_length: int,
        transform=None,
    ) -> None:
        self.image_dir = image_dir
        self.processed_captions = processed_captions
        self.vocab_data = vocab_data
        self.image_names = image_names
        self.max_caption_length = max_caption_length
        self.transform = transform

        self.pad_idx = vocab_data['word2idx']['<PAD>']

        self.samples = []
        for img_name in image_names:
            if img_name in processed_captions:
                for caption in processed_captions[img_name]:
                    self.samples.append((img_name, caption))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption_indices = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = caption_indices[:self.max_caption_length]
        caption_len = len(caption)
        padded = caption + [self.pad_idx] * (self.max_caption_length - caption_len)
        caption_tensor = torch.LongTensor(padded)

        return image, caption_tensor, caption_len


def _get_transforms(image_size: int = 224, mode: str = "train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_data_loaders(
    image_dir: str,
    processed_data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
):
    """Return train/val/test dataloaders + vocab metadata."""
    with open(os.path.join(processed_data_dir, 'vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)
    with open(os.path.join(processed_data_dir, 'captions_processed.pkl'), 'rb') as f:
        processed_captions = pickle.load(f)
    with open(os.path.join(processed_data_dir, 'dataset_splits.pkl'), 'rb') as f:
        splits = pickle.load(f)

    max_caption_length = vocab_data['max_caption_length']

    train_dataset = ImageCaptionDataset(
        image_dir=image_dir,
        processed_captions=processed_captions,
        vocab_data=vocab_data,
        image_names=splits['train'],
        max_caption_length=max_caption_length,
        transform=_get_transforms(image_size, mode='train'),
    )
    val_dataset = ImageCaptionDataset(
        image_dir=image_dir,
        processed_captions=processed_captions,
        vocab_data=vocab_data,
        image_names=splits['val'],
        max_caption_length=max_caption_length,
        transform=_get_transforms(image_size, mode='val'),
    )
    test_dataset = ImageCaptionDataset(
        image_dir=image_dir,
        processed_captions=processed_captions,
        vocab_data=vocab_data,
        image_names=splits['test'],
        max_caption_length=max_caption_length,
        transform=_get_transforms(image_size, mode='test'),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab_data
