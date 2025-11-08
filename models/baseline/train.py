"""
Training script for CNN-LSTM Image Captioning Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt


class Trainer:
    """Trainer class for image captioning model"""
    
    def __init__(self, model, train_loader, val_loader, vocab_data, config):
        """
        Args:
            model: ImageCaptioningModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab_data: Vocabulary data dictionary
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_data = vocab_data
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_data['word2idx']['<PAD>'])
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, captions, lengths) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, captions)
            
            # Calculate loss (exclude START token from targets)
            targets = captions[:, 1:]  # Remove START token
            outputs = outputs[:, :-1, :]  # Remove last prediction
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                          self.config.get('grad_clip', 5.0))
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader, desc='Validating'):
                # Move to device
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # Forward pass
                outputs = self.model(images, captions)
                
                # Calculate loss
                targets = captions[:, 1:]
                outputs = outputs[:, :-1, :]
                
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'vocab_data': self.vocab_data
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            'checkpoint_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'], 
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def train(self, num_epochs):
        """Train the model for multiple epochs"""
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Save training history
            self.save_training_history()
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = os.path.join(
            self.config['checkpoint_dir'], 
            'training_history.json'
        )
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(
            self.config['checkpoint_dir'], 
            'training_curves.png'
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")


def create_default_config():
    """Create default training configuration"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    import config as cfg
    
    config = {
        'embed_size': cfg.BASELINE['embed_size'],
        'hidden_size': cfg.BASELINE['hidden_size'],
        'num_layers': 1,
        'dropout': cfg.BASELINE['dropout'],
        'train_cnn': False,
        'learning_rate': cfg.BASELINE['learning_rate'],
        'weight_decay': cfg.BASELINE['weight_decay'],
        'grad_clip': cfg.BASELINE['grad_clip'],
        'batch_size': cfg.BASELINE['batch_size'],
        'num_workers': 0,
        'image_size': 224,
        'num_epochs': cfg.BASELINE['num_epochs'],
        'checkpoint_dir': './checkpoints'
    }
    return config


if __name__ == "__main__":
    from model import ImageCaptioningModel
    from dataset import ImageCaptionDataset
    import pickle
    
    # Configuration
    config = create_default_config()
    
    # Paths
    image_dir = "../raw_data/Images"
    processed_data_dir = "../data"
    
    # Load vocabulary
    with open(os.path.join(processed_data_dir, 'vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)
    
    # Create model (for testing)
    model = ImageCaptioningModel(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=vocab_data['vocab_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        train_cnn=config['train_cnn']
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
