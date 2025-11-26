"""
Training entrypoint for the standalone BaselineCaptionModel.

Fully self-contained baseline training pipeline.
"""

from __future__ import annotations

import os

from model_baseline_only import (
    BaselineCaptionModel,
    get_data_loaders,
    Trainer,
    create_default_config,
)


def main() -> None:
    config = create_default_config()

    image_dir = "./raw_data/Images"
    processed_data_dir = "./data"

    print("\nLoading datasets...")
    train_loader, val_loader, _, vocab_data = get_data_loaders(
        image_dir=image_dir,
        processed_data_dir=processed_data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )

    model = BaselineCaptionModel(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=vocab_data['vocab_size'],
        dropout=config['dropout'],
        train_cnn=config['train_cnn']
    )

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    trainer = Trainer(model, train_loader, val_loader, vocab_data, config)
    print("\nStarting baseline-only training...")
    trainer.train(config['num_epochs'])


if __name__ == "__main__":
    main()
