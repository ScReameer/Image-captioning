import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        # for param in resnet.parameters():
        #     param.requires_grad_(False)
        modules = list(resnet.children())[:-2]
        resnet_output_size = 2048
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(
            in_features=resnet_output_size,
            out_features=embed_size
        )

        
    def forward(self, images):
        with torch.no_grad():
            features: torch.Tensor = self.resnet(images)
            # [B, feature_maps, size1, size2] -> [B, flatten_size, feature_maps]
            features = features.flatten(start_dim=-2, end_dim=-1).movedim(-1, 1)
        features = self.linear(features)
        return features