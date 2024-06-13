from .pos_encoder import PositionalEncoding

import torch
import numpy as np
from torch import nn

class Decoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_heads: int, dropout_rate: float) -> None:
        """Decoder with Transformer for image captioning task

        Args:
            `d_model` (`int`): text embedding size and also hidden size of Transformer
            `vocab_size` (`int`): size of vocabulary (total words in text corpus)
            `num_heads` (`int`): heads of Transformer, must be divisible by `d_model` without remainder
            `dropout_rate` (`float`): droupout regularization
        """
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout_p=dropout_rate)
        self.fc = nn.Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask=None) -> torch.Tensor:
        tgt = self.embed(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        x = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        x = self.fc(x)
        return x
    
