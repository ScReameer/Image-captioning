from .net_modules.decoder import Decoder
from .net_modules.encoder import Encoder
from torch import nn
import torch
from torch import optim
import lightning as L

class Model(L.LightningModule):
    def __init__(self, vocab_size, d_model, num_heads, dropout_rate=0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.save_hyperparameters(dict(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        ))
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size
        )
        self.class_weights = torch.ones((vocab_size,), device=self.device)
        self.class_weights[0] = torch.tensor(.0) # <PAD>
        self.class_weights[3] = torch.tensor(.0) # <UNK>
        
       
    def training_step(self, batch, batch_idx):
        self.train()
        imgs, captions = batch 
        captions_input = captions[:, :-1]
        captions_expected = captions[:, 1:]
        sequence_length = captions_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)
        predicted = self.forward(imgs=imgs, captions=captions_input, tgt_mask=tgt_mask) # [B, seq, vocab_size]
        loss: torch.Tensor = nn.functional.cross_entropy(
            predicted.contiguous().view(-1, self.vocab_size), # [B*seq, vocab_size]
            captions_expected.contiguous().view(-1), # [B*seq]
            ignore_index=0, # <PAD>
        )
        self.log('train_CE', loss, prog_bar=True, logger=self.logger, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            imgs, captions = batch 
            captions_input = captions[:, :-1]
            captions_expected = captions[:, 1:]
            sequence_length = captions_input.size(1)
            tgt_mask = self.get_tgt_mask(sequence_length)
            predicted = self.forward(imgs=imgs, captions=captions_input, tgt_mask=tgt_mask)
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size),
                captions_expected.contiguous().view(-1),
                weight=self.class_weights.to(predicted.device),
            )
            self.log('val_CE', loss, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
            return loss
        
    def test_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            imgs, captions = batch 
            captions_input = captions[:, :-1] # [B, seq]
            captions_expected = captions[:, 1:]
            sequence_length = captions_input.size(1)
            tgt_mask = self.get_tgt_mask(sequence_length)
            predicted = self.forward(imgs=imgs, captions=captions_input, tgt_mask=tgt_mask)
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size),
                captions_expected.contiguous().view(-1),
                weight=self.class_weights.to(predicted.device),
            )
            self.log('test_CE', loss, prog_bar=True, logger=None)
            return loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def forward(self, imgs, captions, tgt_mask):
        features = self.encoder(imgs)
        predicted = self.decoder(src=features, tgt=captions, tgt_mask=tgt_mask)
        return predicted
    
    def predict(model, image, max_length=50, SOS_token=1, EOS_token=2):
        device = image.device
        model.eval().to(device)
        image = image.unsqueeze(0)
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
            with torch.no_grad():
                pred: torch.Tensor = model(image, y_input, tgt_mask)
            
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