import torch
from torch import nn
from .attention import CausalAttention, CrossAttention
from .encoder import Encoder

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout_rate) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(inplace=True),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate)
        )
        self.layernorm = nn.LayerNorm(normalized_shape=d_model)
        
    def forward(self, x):
        x = self.layernorm(x + self.seq(x))
        return x
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout_rate) -> None:
        super().__init__()
        self.causal_attn = CausalAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.cross_attn = CrossAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)
    
    def forward(self, context, x):
        x = self.causal_attn(x)
        x = self.cross_attn(context=context, x=x)
        x = self.ffn(x)
        self.last_attn_scores = self.cross_attn.last_attn_scores
        return x
    
class Decoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, dff, num_heads, dropout_rate) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.Sequential(*[DecoderLayer(d_model, dff, num_heads, dropout_rate) for _ in range(num_layers)])
        self.last_attn_scores = None
        
    def forward(self, context, x):
        context = context
        x = self.embedding(x)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](context=context, x=x)
        self.last_attn_scores = self.decoder_layers[-1].last_attn_scores
        return x
    
class Model(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, dff, num_heads, dropout_rate=0.5) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(
            num_layers=num_layers,
            vocab_size=vocab_size,
            d_model=d_model,
            dff=dff,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.fc = nn.Linear(in_features=d_model, out_features=vocab_size)
        
    def forward(self, imgs, captions):
        features = self.encoder(imgs)
        predicted = self.decoder(context=features, x=captions)
        return self.fc(predicted)
    
    def caption_image(self, img, max_length=20):
        img = img.unsqueeze(0)
        start_token = torch.tensor([[1]], device=img.device)

        for i in range(max_length):
            if i == 0:
                preds = self(img, start_token)
            else:
                preds = self(img, start_token)[:, -1, :].unsqueeze(1)
            pred_idx = preds.argmax(-1)
            start_token = torch.cat([start_token, pred_idx], dim=1)
            if pred_idx.item() == 2:
                break
            
        return start_token.cpu().numpy().squeeze()