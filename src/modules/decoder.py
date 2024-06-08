import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
  
class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_heads, dropout_rate) -> None:
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(dim_model=d_model, dropout_p=dropout_rate, max_len=5000)
        self.fc = nn.Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask=None, tgt_pad_mask=None):
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        x = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        x = self.fc(x)
        return x
    
