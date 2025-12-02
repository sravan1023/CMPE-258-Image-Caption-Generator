
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TrainerAttention:
    
    def __init__(self, model, train_loader, val_loader, vocab_data, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_data = vocab_data
        self.config = config
        
        # Using MPS
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Success: Using Apple MPS (Metal Performance Shaders) acceleration.")
        else:
            self.device = torch.device('cpu')
            print("Warning: Using CPU. Training will be slow.")
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with Differential Learning Rates 
        if config.get('fine_tune_encoder', False):
            # For fine-tuning, we give the encoder a much lower learning rate
            encoder_params = list(map(id, model.encoder.parameters()))
            decoder_params = filter(lambda p: id(p) not in encoder_params, model.parameters())
            
            self.optimizer = optim.Adam([
                {'params': model.encoder.parameters(), 'lr': config.get('encoder_lr', 1e-4)},
                {'params': decoder_params, 'lr': config['learning_rate']}
            ], weight_decay=config.get('weight_decay', 0))
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0)
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=2
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.alpha_c = config.get('alpha_c', 1.0)
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    def train_epoch(self, epoch):
        # Epoch 1-5: Freeze Encoder (Fast, trains LSTM only)
        # Epoch 6+: Unfreeze Encoder (Slow, trains everything for accuracy)
        # This is to make the run time lower and also achieve less loss value.
        fine_tune = self.config.get('fine_tune_encoder', False) and (epoch > 5)
        
        if fine_tune:
            print(f"Epoch {epoch}: Encoder UN-FROZEN (Fine-Tuning Mode)")
            self.model.encoder.train()
            for p in self.model.encoder.parameters():
                p.requires_grad = True
        else:
            print(f"Epoch {epoch}: Encoder FROZEN (Warm-up Mode)")
            self.model.encoder.eval()
            for p in self.model.encoder.parameters():
                p.requires_grad = False
                
        self.model.decoder.train() # Decoder always trains
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, captions, lengths) in enumerate(progress_bar):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lengths = lengths.unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, alphas, encoded_captions, decode_lengths, sort_ind = self.model(
                images, captions, caption_lengths
            )

            targets = encoded_captions[:, 1:]

            predictions_packed = pack_padded_sequence(
                predictions, 
                decode_lengths, 
                batch_first=True
            )
            
            targets_packed = pack_padded_sequence(
                targets, 
                decode_lengths, 
                batch_first=True
            )
            
            # Calculate Cross Entropy Loss
            loss = self.criterion(predictions_packed.data, targets_packed.data)
            
            # Add Doubly Stochastic Attention Regularization
            # Helps the model look at every part of the image exactly once, alpha_c=1
            alpha_reg = self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss += alpha_reg
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 5.0))
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lengths = lengths.unsqueeze(1).to(self.device)
                
                predictions, alphas, encoded_captions, decode_lengths, sort_ind = self.model(
                    images, captions, caption_lengths
                )

                targets = encoded_captions[:, 1:]

                predictions_packed = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
                targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)
                
                loss = self.criterion(predictions_packed.data, targets_packed.data)
                
                alpha_reg = self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
                loss += alpha_reg
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'vocab_data': self.vocab_data
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], 'checkpoint_primary_model_latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], 'checkpoint_primary_model_best.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def train(self, num_epochs):
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print("-" * 50)
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            self.scheduler.step(val_loss)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            self.plot_training_curves()
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
    
    def plot_training_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['checkpoint_dir'], 'training_curves_primary_model.png'))
        plt.close()


def create_attention_config():
    # Added all the hyperparameters in config.py
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    import config as cfg
    
    config = {
        'embed_size': cfg.ATTENTION['embed_size'],
        'attention_dim': cfg.ATTENTION['attention_dim'],
        'decoder_dim': cfg.ATTENTION['decoder_dim'],
        'encoder_dim': cfg.ATTENTION['encoder_dim'],
        'dropout': cfg.ATTENTION['dropout'],
        
        'fine_tune_encoder': cfg.ATTENTION['fine_tune_encoder'],
        'encoder_lr': cfg.ATTENTION['encoder_lr'],
        'learning_rate': cfg.ATTENTION['learning_rate'],
        
        'weight_decay': cfg.ATTENTION['weight_decay'],
        'grad_clip': cfg.ATTENTION['grad_clip'],
        'alpha_c': cfg.ATTENTION['alpha_c'],
        
        'batch_size': cfg.ATTENTION['batch_size'],
        'num_workers': cfg.ATTENTION.get('num_workers', 4),
        'image_size': 224,
        'num_epochs': cfg.ATTENTION['num_epochs'],
        'checkpoint_dir': './checkpoints'
    }
    return config


if __name__ == "__main__":
    from .model import ImageCaptioningModelAttention
    from .dataset import get_data_loaders_attention
    
    # Configuration
    config = create_attention_config()
    
    # Paths
    image_dir = "./raw_data/Images" 
    processed_data_dir = "./data"
    
    # Load vocabulary
    with open(os.path.join(processed_data_dir, 'vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, vocab_data = get_data_loaders_attention(
        image_dir=image_dir,
        processed_data_dir=processed_data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    # Create model
    model = ImageCaptioningModelAttention(
        embed_size=config['embed_size'],
        attention_dim=config['attention_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=vocab_data['vocab_size'],
        encoder_dim=config['encoder_dim'],
        dropout=config['dropout']
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = TrainerAttention(model, train_loader, val_loader, vocab_data, config)
    
    # Start training
    print("\nStarting training...")
    trainer.train(config['num_epochs'])

    print("\nTraining finished.")