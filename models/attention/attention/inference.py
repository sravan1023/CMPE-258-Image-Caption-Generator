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
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

class CaptionGeneratorAttention:
    """Generate captions for images with attention visualization"""
    
    def __init__(self, model, vocab_data, device='cpu'):
        self.model = model
        self.vocab_data = vocab_data
        self.device = device
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
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def generate_caption(self, image_path, max_length=20):
        """Generate caption for an image with attention weights"""
        image = self.load_image(image_path)
        
        with torch.no_grad():
            caption_indices, alphas = self.model.generate_caption(
                image, 
                max_length=max_length,
                start_token=self.start_token,
                end_token=self.end_token
            )
        
        caption_indices = caption_indices.cpu().numpy().tolist()
        
        # --- DEBUG PRINT ---
        # This will show you the raw numbers. 
        # If you see numbers like [1, 45, 239, 2], the model is working!
        # If you see [1, 0, 0, 0], then the model is actually predicting UNK.
        print(f"DEBUG: Predicted Indices: {caption_indices}")
        # -------------------

        caption = self.indices_to_caption(caption_indices)
        attention_weights = alphas.cpu().numpy()
        
        return caption, caption_indices, attention_weights
    
    def indices_to_caption(self, indices):
        """Convert indices to caption string (Robust Fix)"""
        words = []
        for idx in indices:
            if idx == self.end_token:
                break
            if idx not in [self.pad_token, self.start_token]:
                # --- FIX: Try Integer key first, then String key ---
                if idx in self.idx2word:
                    word = self.idx2word[idx]
                elif str(idx) in self.idx2word:
                    word = self.idx2word[str(idx)]
                else:
                    word = '<UNK>'
                # ---------------------------------------------------
                words.append(word)
        return ' '.join(words)
    
    def visualize_attention(self, image_path, save_path=None, smooth=True):
        """Generate caption and visualize attention weights"""
        caption, caption_indices, attention_weights = self.generate_caption(image_path)
        
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        words = caption.split()
        num_words = min(len(words), 9)
        
        if num_words == 0:
            print("No words generated!")
            return
        
        cols = 3
        rows = (num_words + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_words):
            row = idx // cols
            col = idx % cols
            if row >= rows or col >= cols: continue
                
            ax = axes[row, col]
            alpha = attention_weights[idx].reshape(7, 7)
            
            if smooth:
                alpha = skimage.transform.pyramid_expand(alpha, upscale=32, sigma=8)
            else:
                alpha = skimage.transform.resize(alpha, (224, 224))
            
            alpha_resized = skimage.transform.resize(alpha, (image_np.shape[0], image_np.shape[1]))
            
            ax.imshow(image_np)
            ax.imshow(alpha_resized, alpha=0.7, cmap='jet')
            ax.set_title(f'"{words[idx]}"', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        for idx in range(num_words, rows * cols):
            row = idx // cols
            col = idx % cols
            if row < rows and col < cols:
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
        """Generate and visualize caption (simple version)"""
        caption, _, _ = self.generate_caption(image_path)
        image = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.title(f'Generated Caption:\n{caption}', fontsize=14, wrap=True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close()
        return caption


def load_model_from_checkpoint_attention(checkpoint_path, device='cpu'):
    """Load attention model from checkpoint"""
    from .model import ImageCaptioningModelAttention
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    vocab_data = checkpoint['vocab_data']
    
    model = ImageCaptioningModelAttention(
        embed_size=config['embed_size'],
        attention_dim=config['attention_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=vocab_data['vocab_size'],
        encoder_dim=config['encoder_dim'],
        dropout=config['dropout'],
        train_cnn=config.get('train_cnn', False) 
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded attention model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab_data, config


if __name__ == "__main__":
    import argparse
    import random

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # --- Arguments ---
    parser = argparse.ArgumentParser(description='Generate Caption for an Image')
    parser.add_argument('--image', type=str, help='Path to a specific image file (optional)')
    args = parser.parse_args()

    checkpoint_path = "./checkpoints/checkpoint_primary_model_best.pth"
    default_image_dir = "./raw_data/Images" 
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Load model once
    model, vocab_data, config = load_model_from_checkpoint_attention(checkpoint_path, device)
    generator = CaptionGeneratorAttention(model, vocab_data, device)
    
    # Determine which image to use
    target_image_path = ""
    
    if args.image:
        # Case A: User provided a specific image path
        target_image_path = args.image
        if not os.path.exists(target_image_path):
            print(f"Error: File not found at {target_image_path}")
            sys.exit(1)
    else:
        # Case B: Pick a random image from the folder
        if os.path.exists(default_image_dir):
            all_images = [f for f in os.listdir(default_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if all_images:
                random_image = random.choice(all_images)
                target_image_path = os.path.join(default_image_dir, random_image)
            else:
                print("No images found in default directory.")
                sys.exit(1)
        else:
            print("Default image directory not found.")
            sys.exit(1)

    print(f"\nGenerating caption for: {target_image_path}")
    
    caption = generator.visualize_prediction(target_image_path, save_path="./sample_prediction.png")
    print(f"Generated Caption: {caption}")
    
    print("Generating attention visualization...")
    generator.visualize_attention(target_image_path, save_path="./sample_attention_map.png")
    print("Done! Check 'sample_prediction.png' and 'sample_attention_map.png'")