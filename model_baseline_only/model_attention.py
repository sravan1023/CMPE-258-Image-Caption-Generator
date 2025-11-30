"""
Attention-based image captioning model.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderAttention(nn.Module):
    """CNN encoder that outputs spatial features for attention."""

    def __init__(self, encoder_dim: int = 256, train_cnn: bool = False) -> None:
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.conv_dim = 512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.project = nn.Linear(self.conv_dim, encoder_dim)
        self.encoder_dim = encoder_dim

        if not train_cnn:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.adaptive_pool(features)
        batch_size, channels, h, w = features.size()
        features = features.permute(0, 2, 3, 1).view(batch_size, -1, channels)
        features = self.project(features)
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class DecoderAttention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        attention_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        caption_lengths = caption_lengths.squeeze(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()

        max_decode_len = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_decode_len, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_decode_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h[:batch_size_t], c[:batch_size_t] = self.decode_step(
                lstm_input, (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h[:batch_size_t]))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, encoded_captions, decode_lengths, sort_ind

    def sample(self, encoder_out, max_length, start_token, end_token):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        h, c = self.init_hidden_state(encoder_out)
        inputs = torch.LongTensor([start_token] * batch_size).to(encoder_out.device)

        captions = []
        alphas = []

        for _ in range(max_length):
            embeddings = self.embedding(inputs)
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))
            scores = self.fc(h)
            predicted = scores.argmax(dim=1)

            captions.append(predicted)
            alphas.append(alpha)
            inputs = predicted

        captions = torch.stack(captions, dim=1)
        alphas = torch.stack(alphas, dim=1)
        return captions, alphas


class AttentionCaptionModel(nn.Module):
    def __init__(
        self,
        embed_size: int,
        attention_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 256,
        dropout: float = 0.5,
        train_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EncoderAttention(encoder_dim, train_cnn=train_cnn)
        self.decoder = DecoderAttention(
            embed_size=embed_size,
            attention_dim=attention_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions, caption_lengths)
        return outputs

    def generate_caption(self, images, max_length, start_token, end_token):
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(images)
            captions, alphas = self.decoder.sample(encoder_out, max_length, start_token, end_token)
        return captions, alphas
