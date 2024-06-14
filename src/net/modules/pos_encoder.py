import torch
import numpy as np
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_p: float, max_len=10000):
        """Positional Encoding for Transformer
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Args:
            `d_model` (`int`): text embedding size
            `dropout_p` (`float`): dropout regularization
            `max_len` (`int`, optional): `max_len` determines how far the position can have an effect on a token (window). Defaults to `10000`.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)) / d_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])