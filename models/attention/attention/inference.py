"""
Inference and Evaluation script for Image Captioning Model with Attention
Includes attention visualization
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import skimage.transform


class CaptionGeneratorAttention:
    """Generate captions for images with attention visualization"""
    
    def __init__(self, model, vocab_data, device='cuda'):
        """
        Args:
            model: Trained ImageCaptioningModelAttention
            vocab_data: Vocabulary data dictionary
            device: Device to run inference on
        """
        self.model = model
        self.vocab_data = vocab_data
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.start_token = self.word2idx['<START>']
        self.end_token = self.word2idx['<END>']
        self.pad_token = self.word2idx['<PAD>']
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # type: ignore[union-attr]
        return image_tensor.to(self.device)
    
    def generate_caption(self, image_path, max_length=20):
        """
        Generate caption for an image with attention weights
        Args:
            image_path: Path to image file
            max_length: Maximum caption length
        Returns:
            caption: Generated caption as string
            caption_indices: Caption as list of indices
            attention_weights: Attention weights for each word
        """
        # Load image
        image = self.load_image(image_path)
        
        # Generate caption
        with torch.no_grad():
            caption_indices, alphas = self.model.generate_caption(
                image, 
                max_length=max_length,
                start_token=self.start_token,
                end_token=self.end_token
            )
        
        # Convert indices to words
        caption_indices = caption_indices.cpu().numpy().tolist()
        caption = self.indices_to_caption(caption_indices)
        
        # Get attention weights
        attention_weights = alphas.cpu().numpy()  # (seq_len, 49)
        
        return caption, caption_indices, attention_weights
    
    def indices_to_caption(self, indices):
        """Convert indices to caption string"""
        words = []
        for idx in indices:
            if idx == self.end_token:
                break
            if idx not in [self.pad_token, self.start_token]:
                word = self.idx2word.get(str(idx), '<UNK>')
                words.append(word)
        return ' '.join(words)
    
    def visualize_attention(self, image_path, save_path=None, smooth=True):
        """
        Generate caption and visualize attention weights
        Args:
            image_path: Path to image
            save_path: Path to save visualization (optional)
            smooth: Whether to smooth attention maps
        """
        # Generate caption with attention
        caption, caption_indices, attention_weights = self.generate_caption(image_path)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get words
        words = caption.split()
        
        # Number of words to visualize (limit to prevent overcrowding)
        num_words = min(len(words), 9)  # 3x3 grid max
        
        if num_words == 0:
            print("No words generated!")
            return
        
        # Create subplot grid
        cols = 3
        rows = (num_words + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Reshape attention to spatial dimensions (7x7)
        for idx in range(num_words):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get attention weights for this word
            alpha = attention_weights[idx].reshape(7, 7)
            
            # Resize to match image size
            if smooth:
                alpha = skimage.transform.pyramid_expand(alpha, upscale=32, sigma=8)
            else:
                alpha = skimage.transform.resize(alpha, (224, 224))
            
            # Resize to original image size
            alpha_resized = skimage.transform.resize(alpha, (image_np.shape[0], image_np.shape[1]))  # type: ignore[arg-type]
            
            # Display image
            ax.imshow(image_np)
            
            # Overlay attention heatmap
            ax.imshow(alpha_resized, alpha=0.7, cmap='jet')
            
            # Set title with word
            ax.set_title(f'"{words[idx]}"', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_words, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Full Caption: {caption}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return caption, attention_weights
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Generate and visualize caption (simple version)
        Args:
            image_path: Path to image
            save_path: Path to save visualization (optional)
        """
        # Generate caption
        caption, _, _ = self.generate_caption(image_path)
        
        # Load and display image
        image = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(image))  # Convert PIL Image to numpy array
        plt.axis('off')
        plt.title(f'Generated Caption:\n{caption}', fontsize=14, wrap=True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return caption


def load_model_from_checkpoint_attention(checkpoint_path, device='cuda'):
    """Load attention model from checkpoint"""
    from model import ImageCaptioningModelAttention
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    vocab_data = checkpoint['vocab_data']
    
    # Create model
    model = ImageCaptioningModelAttention(
        embed_size=config['embed_size'],
        attention_dim=config['attention_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=vocab_data['vocab_size'],
        encoder_dim=config['encoder_dim'],
        dropout=config['dropout'],
        train_cnn=config['train_cnn']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded attention model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab_data, config


if __name__ == "__main__":
    import sys
    
    # Paths
    checkpoint_path = "./checkpoints/checkpoint_primary_model_best.pth"
    image_dir = "../raw_data/Images"
    processed_data_dir = "../data"
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the primary model first!")
        sys.exit(1)
    
    model, vocab_data, config = load_model_from_checkpoint_attention(checkpoint_path, device)
    
    # Create caption generator
    generator = CaptionGeneratorAttention(model, vocab_data, device)
    
    # Example: Generate caption with attention visualization
    test_image = os.path.join(image_dir, "1000268201_693b08cb0e.jpg")
    if os.path.exists(test_image):
        print(f"\nGenerating caption with attention for: {test_image}")
        
        # Simple prediction
        caption = generator.visualize_prediction(
            test_image,
            save_path="./sample_attention_simple.png"
        )
        print(f"Caption: {caption}")
        
        # Attention visualization
        print("\nGenerating attention visualization...")
        generator.visualize_attention(
            test_image,
            save_path="./sample_attention_detailed.png"
        )
        print("Attention visualization complete!")
    else:
        print(f"Test image not found: {test_image}")
