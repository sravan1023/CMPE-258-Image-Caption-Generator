"""
Configuration file for hyperparameter tuning
"""

# Data paths
DATA_DIR = './data'
IMAGE_DIR = './raw_data/Images'
CAPTIONS_FILE = './raw_data/captions.txt'
CHECKPOINT_DIR = './checkpoints'

# Preprocessing
MIN_WORD_FREQ = 2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Baseline model hyperparameters
BASELINE = {
    'embed_size': 256,
    'hidden_size': 512,
    'dropout': 0.6,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 20,
    'weight_decay': 5e-5,
    'grad_clip': 5.0
}

# primary model hyperparameters
ATTENTION = {
    'embed_size': 256,
    'attention_dim': 512,
    'decoder_dim': 512,
    'dropout': 0.5,
    'learning_rate': 0.0004,
    'batch_size': 32,
    'num_epochs': 20,
    'weight_decay': 1e-5,
    'grad_clip': 5.0,
    'alpha_c': 1.0
}


