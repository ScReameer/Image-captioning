import torch
from torch import nn
import math
from .encoder import Encoder

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
    
class Model(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, dropout_rate=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size
        )
        
        
    def forward(self, imgs, captions, tgt_mask):
        features = self.encoder(imgs)
        predicted = self.decoder(src=features, tgt=captions, tgt_mask=tgt_mask)
        return predicted
    
    def predict(model, input_sequence, max_length=20, SOS_token=1, EOS_token=2):
        """
        Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
        Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        """
        model.eval()
        device = input_sequence.device
        input_sequence = input_sequence.unsqueeze(0)
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        num_tokens = len(input_sequence[0])

        for _ in range(max_length):
            # Get source mask
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
            
            pred: torch.Tensor = model(input_sequence, y_input, tgt_mask)
            
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token:
                break

        return y_input.view(-1).tolist()
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token=0) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)