"""
Entrypoint to train the attention-based captioning model.
"""

from __future__ import annotations

from model_baseline_only import (
    AttentionCaptionModel,
    get_data_loaders_attention,
    AttentionTrainer,
    create_attention_config,
)


def main():
    config = create_attention_config()

    image_dir = "./raw_data/Images"
    processed_data_dir = "./data"

    print("\nLoading datasets (attention)...")
    train_loader, val_loader, _, vocab_data = get_data_loaders_attention(
        image_dir=image_dir,
        processed_data_dir=processed_data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size'],
    )

    model = AttentionCaptionModel(
        embed_size=config['embed_size'],
        attention_dim=config['attention_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=vocab_data['vocab_size'],
        encoder_dim=config['encoder_dim'],
        dropout=config['dropout'],
        train_cnn=config['train_cnn'],
    )

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    trainer = AttentionTrainer(model, train_loader, val_loader, vocab_data, config)
    trainer.train(config['num_epochs'])


if __name__ == "__main__":
    main()
