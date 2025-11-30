"""
Trainer for attention-based captioning model.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import config as cfg


class AttentionTrainer:
    def __init__(self, model, train_loader, val_loader, vocab_data, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_data = vocab_data
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_data['word2idx']['<PAD>'])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0),
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        self.alpha_c = config.get('alpha_c', 1.0)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for images, captions, lengths in tqdm(self.train_loader, desc=f'Epoch {epoch}'):
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()
            predictions, alphas, _, decode_lengths, sort_ind = self.model(images, captions, lengths.unsqueeze(1))

            targets = captions[sort_ind][:, 1:]
            predictions = predictions[:, :max(decode_lengths), :]

            packed_predictions = torch.cat(
                [predictions[i, :decode_lengths[i], :] for i in range(predictions.size(0))], dim=0
            )
            packed_targets = torch.cat(
                [targets[i, :decode_lengths[i]] for i in range(targets.size(0))], dim=0
            )

            loss = self.criterion(packed_predictions, packed_targets)
            loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 5.0))
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, captions, lengths in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)

                predictions, alphas, _, decode_lengths, sort_ind = self.model(images, captions, lengths.unsqueeze(1))
                targets = captions[sort_ind][:, 1:]
                predictions = predictions[:, :max(decode_lengths), :]

                packed_predictions = torch.cat(
                    [predictions[i, :decode_lengths[i], :] for i in range(predictions.size(0))], dim=0
                )
                packed_targets = torch.cat(
                    [targets[i, :decode_lengths[i]] for i in range(targets.size(0))], dim=0
                )

                loss = self.criterion(packed_predictions, packed_targets)
                loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'vocab_data': self.vocab_data,
        }
        latest_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint_attention_latest.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint_attention_best.pth')
            torch.save(checkpoint, best_path)

    def train(self, num_epochs: int):
        print(f"Training on device: {self.device}")
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)
            self._save_history()

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        self._plot_curves()

    def _save_history(self):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        history_path = os.path.join(self.config['checkpoint_dir'], 'training_history_attention.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

    def _plot_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Attention Model Training/Validation Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.config['checkpoint_dir'], 'training_curves_attention.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_attention_config():
    return {
        'embed_size': cfg.ATTENTION['embed_size'],
        'attention_dim': cfg.ATTENTION['attention_dim'],
        'decoder_dim': cfg.ATTENTION['decoder_dim'],
        'encoder_dim': 256,
        'dropout': cfg.ATTENTION['dropout'],
        'train_cnn': False,
        'learning_rate': cfg.ATTENTION['learning_rate'],
        'weight_decay': cfg.ATTENTION['weight_decay'],
        'grad_clip': cfg.ATTENTION['grad_clip'],
        'alpha_c': cfg.ATTENTION['alpha_c'],
        'batch_size': cfg.ATTENTION['batch_size'],
        'num_workers': 0,
        'image_size': 224,
        'num_epochs': cfg.ATTENTION['num_epochs'],
        'checkpoint_dir': './checkpoints',
    }
