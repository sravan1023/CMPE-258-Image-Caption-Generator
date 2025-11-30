"""
Baseline-only CNN-LSTM image captioning model.

Implements a ResNet-50 encoder followed by a single-layer LSTM decoder.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineEncoder(nn.Module):
    """CNN encoder that produces a compact embedding per image."""

    def __init__(self, embed_size: int, train_cnn: bool = False) -> None:
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # keep global average pooling, drop FC
        self.backbone = nn.Sequential(*modules)
        self.feature_dim = resnet.fc.in_features  # 2048
        self.project = nn.Linear(self.feature_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

        if not train_cnn:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into embeddings."""
        feats = self.backbone(images)  # (batch, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (batch, 2048)
        feats = self.project(feats)
        feats = self.bn(feats)
        return feats


class BaselineDecoder(nn.Module):
    """Single-layer LSTM decoder that generates captions."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # project encoded image embedding to initial hidden/cell states
        self.init_hidden = nn.Linear(embed_size, hidden_size)
        self.init_cell = nn.Linear(embed_size, hidden_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor, scheduled_sampling_prob: float = 0.0) -> torch.Tensor:
        """
        Forward pass for teacher-forced training.

        Args:
            features: Encoded images (batch, embed_size)
            captions: Tokenized captions (batch, max_len)
            scheduled_sampling_prob: Probability of feeding the model's previous token
        Returns:
            Logits over vocabulary for each time-step.
        """
        inputs = captions[:, :-1]  # remove <END>, keep <START>
        hidden_state, cell_state = self._init_state(features)

        if scheduled_sampling_prob <= 0.0:
            embeddings = self.embedding(inputs)
            outputs, _ = self.lstm(embeddings, (hidden_state, cell_state))
            outputs = self.fc(self.dropout(outputs))
            return outputs

        batch_size, seq_len = inputs.size()
        outputs = []
        prev_tokens = None

        for t in range(seq_len):
            step_tokens = inputs[:, t]
            if t > 0 and prev_tokens is not None:
                mask = (torch.rand(batch_size, device=features.device) < scheduled_sampling_prob)
                if mask.any():
                    step_tokens = step_tokens.clone()
                    step_tokens[mask] = prev_tokens[mask]

            step_embed = self.embedding(step_tokens).unsqueeze(1)
            lstm_out, (hidden_state, cell_state) = self.lstm(step_embed, (hidden_state, cell_state))
            logits = self.fc(self.dropout(lstm_out.squeeze(1)))
            outputs.append(logits)
            prev_tokens = logits.argmax(dim=1)

        return torch.stack(outputs, dim=1)

    def _init_state(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the initial LSTM states from encoded image features."""
        hidden = torch.tanh(self.init_hidden(features)).unsqueeze(0)
        cell = torch.tanh(self.init_cell(features)).unsqueeze(0)
        return hidden, cell

    def greedy_search(
        self,
        features: torch.Tensor,
        max_length: int,
        start_token: int,
        end_token: int,
    ) -> torch.Tensor:
        """
        Generate captions with greedy decoding.

        Args:
            features: Encoded images (batch, embed_size)
            max_length: Maximum caption length
            start_token: Index of <START>
            end_token: Index of <END>
        Returns:
            Caption indices (batch, max_length)
        """
        batch_size = features.size(0)
        hidden, cell = self._init_state(features)
        inputs = torch.full((batch_size,), start_token, dtype=torch.long, device=features.device)

        samples = []
        for _ in range(max_length):
            embeddings = self.embedding(inputs).unsqueeze(1)
            lstm_out, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
            logits = self.fc(self.dropout(lstm_out.squeeze(1)))
            predicted = logits.argmax(dim=1)
            samples.append(predicted)
            inputs = predicted

        return torch.stack(samples, dim=1)

    def beam_search(
        self,
        features: torch.Tensor,
        max_length: int,
        start_token: int,
        end_token: int,
        beam_width: int = 3,
    ) -> torch.Tensor:
        """
        Beam search decoding for a batch of images.
        Processes each image independently to keep implementation simple.
        """
        device = features.device
        captions = []
        for feature in features:
            hidden, cell = self._init_state(feature.unsqueeze(0))
            sequences = [([start_token], 0.0, hidden, cell)]

            for _ in range(max_length):
                all_candidates = []
                for seq, score, h, c in sequences:
                    if seq[-1] == end_token:
                        all_candidates.append((seq, score, h, c))
                        continue
                    inputs = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                    embeddings = self.embedding(inputs).unsqueeze(1)
                    lstm_out, (h_new, c_new) = self.lstm(embeddings, (h, c))
                    logits = self.fc(self.dropout(lstm_out.squeeze(1)))
                    log_probs = torch.log_softmax(logits, dim=1)
                    top_log_probs, top_idx = log_probs.topk(beam_width, dim=1)
                    for prob, idx in zip(top_log_probs[0], top_idx[0]):
                        candidate = (seq + [idx.item()], score + prob.item(), h_new, c_new)
                        all_candidates.append(candidate)

                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                sequences = ordered[:beam_width]

            best_seq = max(sequences, key=lambda tup: tup[1])[0]
            captions.append(torch.tensor(best_seq[1:], device=device))  # drop start token

        padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=end_token)
        return padded


class BaselineCaptionModel(nn.Module):
    """Full baseline image captioning model."""

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        dropout: float = 0.5,
        train_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = BaselineEncoder(embed_size, train_cnn=train_cnn)
        self.decoder = BaselineDecoder(embed_size, hidden_size, vocab_size, dropout=dropout)

    def forward(self, images: torch.Tensor, captions: torch.Tensor, scheduled_sampling_prob: float = 0.0) -> torch.Tensor:
        features = self.encoder(images)
        outputs = self.decoder(features, captions, scheduled_sampling_prob=scheduled_sampling_prob)
        return outputs

    def generate(
        self,
        images: torch.Tensor,
        max_length: int,
        start_token: int,
        end_token: int,
        method: str = "greedy",
        beam_width: int = 3,
    ) -> torch.Tensor:
        """
        Generate captions for a batch of images.

        Args:
            images: Tensor of shape (batch, 3, H, W)
            max_length: Maximum caption length
            start_token: Index of <START>
            end_token: Index of <END>
            method: 'greedy' or 'beam'
            beam_width: Beam width for beam search
        Returns:
            Tensor of caption indices (batch, max_length)
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(images)
            if method == "beam":
                return self.decoder.beam_search(features, max_length, start_token, end_token, beam_width)
            return self.decoder.greedy_search(features, max_length, start_token, end_token)
