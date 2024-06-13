from ..data_processing.vocabulary import Vocabulary
from .decoder import Decoder
from .encoder import Encoder

import torch
import lightning as L
from torch import nn, optim

class Model(L.LightningModule):
    def __init__(
        self, 
        vocab: Vocabulary,
        d_model: int, 
        num_heads: int,
        lr_start: float,
        gamma: float,
        dropout_rate=0.1
    ) -> None:
        """Model class for image captioning task

        Args:
            `vocab` (`Vocabulary`): vocabulary instance of `src.data_processing.vocabulary.Vocabulary`
            `d_model` (`int`): text embedding size and also hidden size of Transformer
            `num_heads` (`int`): heads of Transformer, must be divisible by `d_model` without remainder
            `lr_start` (`float`): starting learning rate
            `gamma` (`float`): gamma for exponential learning rate scheduler
            `dropout_rate` (`float`, optional): droupout regularization. Defaults to 0.1.
        """
        super().__init__()
        self.vocab = vocab
        self.pad_idx = self.vocab.word2idx['<PAD>']
        self.sos_idx = self.vocab.word2idx['<SOS>']
        self.eos_idx = self.vocab.word2idx['<EOS>']
        self.unk_idx = self.vocab.word2idx['<UNK>']
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.lr_start = lr_start
        self.gamma = gamma
        self.save_hyperparameters(dict(
            vocab_size=self.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            lr_start=self.lr_start,
            gamma=self.gamma
        ))
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            vocab_size=self.vocab_size
        )
        self.class_weights = torch.ones((self.vocab_size, ), device=self.device)
        self.class_weights[self.pad_idx] = torch.tensor(.0)
        self.class_weights[self.unk_idx] = torch.tensor(.0)
        
       
    def training_step(self, batch, batch_idx) -> torch.Tensor:
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
            ignore_index=self.pad_idx,
        )
        self.log('train_CE', loss, prog_bar=True, logger=self.logger, on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
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
        
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, captions = batch 
            captions_input = captions[:, :-1] # [B, seq-1]
            captions_expected = captions[:, 1:] # [B, seq-1]
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
        
    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=self.lr_start)
        exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return {
            'optimizer': optimizer,
            'lr_scheduler': exp_scheduler
        }
    
    def forward(self, imgs, captions, tgt_mask) -> torch.Tensor:
        features = self.encoder(imgs)
        predicted = self.decoder(src=features, tgt=captions, tgt_mask=tgt_mask)
        return predicted
    
    def predict(self, image: torch.Tensor, max_length=50) -> str:
        """Predict caption to image

        Args:
            `image` (`Tensor`): preprocessed (resized and normalized) image of shape `[C, H, W]`
            `max_length` (`int`, optional): max output sentence length. Defaults to `50`.

        Returns:
            `caption` (`str`): predicted caption for image
        """
        device = image.device
        self.eval().to(device)
        image = image.unsqueeze(0)
        y_input = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.get_tgt_mask(y_input.size(1)).to(device)
            with torch.no_grad():
                pred: torch.Tensor = self(image, y_input, tgt_mask)
            next_item = pred.argmax(-1)[0, -1].item()
            next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == self.eos_idx:
                break
        result = y_input.view(-1).tolist()[1:-1]
        return ' '.join([self.vocab.idx2word[idx] for idx in result])
    
    def get_tgt_mask(self, size: int) -> torch.Tensor:
        """Generates a square matrix where the each row allows one word more to be seen

        Args:
            `size` (`int`): sequence length of target, for example if target have shape `[B, S]` then `size = S`

        Returns:
            `mask` (`torch.Tensor`): target mask for transformer
        """
        mask = torch.tril(torch.ones(size, size) == 1).float() # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask