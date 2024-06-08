import torch
from torch import nn

class BaseAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(batch_first=True, **kwargs)
        self.layernorm = nn.LayerNorm(normalized_shape=kwargs['embed_dim'])
        
class CausalAttention(BaseAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(-2), device=x.device)
        attn_output, _ = self.mha(query=x, key=x, value=x, is_causal=True, attn_mask=attn_mask)
        x = self.layernorm(x + attn_output)
        return x
    
class CrossAttention(BaseAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, context, x):
        attn_output, attn_scores = self.mha(query=x, key=context, value=context)
        self.last_attn_scores = attn_scores
        x = self.layernorm(x + attn_output)
        return x