from .decoder import Decoder
from .encoder import Encoder
from torch import nn
import torch

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
            with torch.no_grad():
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