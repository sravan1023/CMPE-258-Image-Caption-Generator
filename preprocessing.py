"""
Data preprocessing for image captioning
Handles caption cleaning, tokenization, and vocabulary building
"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter
import re
import os


class CaptionPreprocessor:
    """Preprocess captions: clean, tokenize, and build vocabulary"""
    
    def __init__(self, captions_file, min_word_freq=5):
        """
        Args:
            captions_file: Path to captions.txt file
            min_word_freq: Minimum frequency for a word to be included in vocabulary
        """
        self.captions_file = captions_file
        self.min_word_freq = min_word_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.max_caption_length = 0
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        
    def clean_caption(self, caption):
        """Clean a single caption"""
        # Convert to lowercase
        caption = caption.lower()
        # Remove special characters and digits
        caption = re.sub(r'[^a-z\s]', '', caption)
        # Remove extra whitespace
        caption = ' '.join(caption.split())
        return caption
    
    def load_and_clean_captions(self):
        """Load captions from file and clean them"""
        df = pd.read_csv(self.captions_file)
        
        # Clean all captions
        df['caption'] = df['caption'].apply(self.clean_caption)
        
        # Group captions by image
        captions_dict = df.groupby('image')['caption'].apply(list).to_dict()
        
        print(f"Loaded {len(captions_dict)} images with {len(df)} total captions")
        return captions_dict
    
    def build_vocabulary(self, captions_dict):
        """Build vocabulary from captions"""
        # Count word frequencies
        word_freq = Counter()
        all_captions = []
        
        for img_captions in captions_dict.values():
            for caption in img_captions:
                words = caption.split()
                word_freq.update(words)
                all_captions.append(words)
        
        # Filter words by minimum frequency
        vocab_words = [word for word, freq in word_freq.items() 
                      if freq >= self.min_word_freq]
        
        # Create word-to-index mapping (reserve indices 0-3 for special tokens)
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        
        for idx, word in enumerate(sorted(vocab_words), start=4):
            self.word2idx[word] = idx
        
        # Create index-to-word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Calculate max caption length
        caption_lengths = [len(caption) for caption in all_captions]
        self.max_caption_length = max(caption_lengths) + 2  # +2 for START and END tokens
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max caption length: {self.max_caption_length}")
        print(f"Average caption length: {np.mean(caption_lengths):.2f}")
        
        return self.word2idx, self.idx2word
    
    def caption_to_indices(self, caption):
        """Convert caption to sequence of indices"""
        words = caption.split()
        indices = [self.word2idx[self.start_token]]
        
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.unk_token])
        
        indices.append(self.word2idx[self.end_token])
        return indices
    
    def indices_to_caption(self, indices):
        """Convert sequence of indices back to caption"""
        words = []
        for idx in indices:
            if idx == self.word2idx[self.end_token]:
                break
            if idx not in [self.word2idx[self.pad_token], self.word2idx[self.start_token]]:
                words.append(self.idx2word[idx])
        return ' '.join(words)
    
    def process_and_save(self, output_dir):
        """Process captions and save preprocessed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and clean captions
        captions_dict = self.load_and_clean_captions()
        
        # Build vocabulary
        self.build_vocabulary(captions_dict)
        
        # Convert all captions to indices
        processed_captions = {}
        for img_name, img_captions in captions_dict.items():
            processed_captions[img_name] = [
                self.caption_to_indices(caption) for caption in img_captions
            ]
        
        # Save processed data
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'max_caption_length': self.max_caption_length
        }
        
        with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab_data, f)
        
        with open(os.path.join(output_dir, 'captions_processed.pkl'), 'wb') as f:
            pickle.dump(processed_captions, f)
        
        # Save original captions dict for reference
        with open(os.path.join(output_dir, 'captions_dict.pkl'), 'wb') as f:
            pickle.dump(captions_dict, f)
        
        print(f"\nPreprocessed data saved to {output_dir}")
        return vocab_data, processed_captions


def split_dataset(captions_dict, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train, validation, and test sets"""
    image_names = list(captions_dict.keys())
    np.random.seed(42)
    np.random.shuffle(image_names)
    
    n_images = len(image_names)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    train_images = image_names[:n_train]
    val_images = image_names[n_train:n_train + n_val]
    test_images = image_names[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")
    
    return train_images, val_images, test_images


if __name__ == "__main__":
    # Example usage
    captions_file = "./raw_data/captions.txt"
    output_dir = "./data"
    
    preprocessor = CaptionPreprocessor(captions_file, min_word_freq=5)
    vocab_data, processed_captions = preprocessor.process_and_save(output_dir)
    
    # Split dataset
    train_imgs, val_imgs, test_imgs = split_dataset(processed_captions)
    
    # Save splits
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    with open(os.path.join(output_dir, 'dataset_splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
    
    print("\nPreprocessing complete!")
