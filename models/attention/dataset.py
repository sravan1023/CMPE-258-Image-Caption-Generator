"""
PyTorch Dataset class for image captioning with attention
Returns caption lengths needed for attention mechanism
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pickle
import numpy as np


class ImageCaptionDatasetAttention(Dataset):
    """Dataset for loading images and their captions (for attention model)"""
    
    def __init__(self, image_dir, processed_captions, vocab_data, 
                 image_names, max_caption_length, transform=None):
        """
        Args:
            image_dir: Directory containing images
            processed_captions: Dictionary of image_name -> list of caption indices
            vocab_data: Dictionary with vocabulary information
            image_names: List of image names to include in this dataset
            max_caption_length: Maximum length for padding captions
            transform: Image transformations
        """
        self.image_dir = image_dir
        self.processed_captions = processed_captions
        self.vocab_data = vocab_data
        self.image_names = image_names
        self.max_caption_length = max_caption_length
        self.transform = transform
        
        self.pad_idx = vocab_data['word2idx']['<PAD>']
        
        # Create a flattened list of (image, caption) pairs
        self.samples = []
        for img_name in image_names:
            if img_name in processed_captions:
                for caption_indices in processed_captions[img_name]:
                    self.samples.append((img_name, caption_indices))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single image-caption pair"""
        img_name, caption_indices = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Pad caption to max_caption_length
        caption = caption_indices[:self.max_caption_length]  # Truncate if too long
        caption_len = len(caption)
        
        # Pad with PAD token
        padded_caption = caption + [self.pad_idx] * (self.max_caption_length - caption_len)
        
        # Convert to tensors
        caption_tensor = torch.LongTensor(padded_caption)
        
        # Return image, caption, and actual caption length (not padded length)
        return image, caption_tensor, torch.LongTensor([caption_len])


def get_transforms_attention(image_size=224, mode='train'):
    """Get image transformations"""
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def collate_fn_attention(data):
    """
    Custom collate function for DataLoader with attention
    Sorts batch by caption length (descending) as required by pack_padded_sequence
    
    Args:
        data: List of tuples (image, caption, length)
    Returns:
        images: Tensor of images (batch_size, 3, H, W)
        captions: Tensor of captions (batch_size, max_length)
        lengths: Tensor of caption lengths (batch_size,)
    """
    # Sort by length (descending)
    data.sort(key=lambda x: x[2].item(), reverse=True)
    
    images, captions, lengths = zip(*data)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Stack captions
    captions = torch.stack(captions, 0)
    
    # Stack lengths and flatten
    lengths = torch.cat(lengths, 0)
    
    return images, captions, lengths


def get_data_loaders_attention(image_dir, processed_data_dir, batch_size=32, 
                                num_workers=0, image_size=224):
    """Create data loaders for train, validation, and test sets (attention model)"""
    
    # Load preprocessed data
    with open(os.path.join(processed_data_dir, 'vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)
    
    with open(os.path.join(processed_data_dir, 'captions_processed.pkl'), 'rb') as f:
        processed_captions = pickle.load(f)
    
    with open(os.path.join(processed_data_dir, 'dataset_splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    
    max_caption_length = vocab_data['max_caption_length']
    
    # Create datasets
    train_dataset = ImageCaptionDatasetAttention(
        image_dir=image_dir,
        processed_captions=processed_captions,
        vocab_data=vocab_data,
        image_names=splits['train'],
        max_caption_length=max_caption_length,
        transform=get_transforms_attention(image_size, mode='train')
    )
    
    val_dataset = ImageCaptionDatasetAttention(
        image_dir=image_dir,
        processed_captions=processed_captions,
        vocab_data=vocab_data,
        image_names=splits['val'],
        max_caption_length=max_caption_length,
        transform=get_transforms_attention(image_size, mode='val')
    )
    
    test_dataset = ImageCaptionDatasetAttention(
        image_dir=image_dir,
        processed_captions=processed_captions,
        vocab_data=vocab_data,
        image_names=splits['test'],
        max_caption_length=max_caption_length,
        transform=get_transforms_attention(image_size, mode='test')
    )
    
    # Create data loaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_attention,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_attention,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_attention,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nData loaders created (attention):")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, vocab_data


if __name__ == "__main__":
    # Test the dataset
    image_dir = "../raw_data/Images"
    processed_data_dir = "../data/processed"
    
    train_loader, val_loader, test_loader, vocab_data = get_data_loaders_attention(
        image_dir, processed_data_dir, batch_size=4
    )
    
    # Test loading a batch
    images, captions, lengths = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"Images shape: {images.shape}")
    print(f"Captions shape: {captions.shape}")
    print(f"Lengths shape: {lengths.shape}")
    print(f"Lengths: {lengths}")
