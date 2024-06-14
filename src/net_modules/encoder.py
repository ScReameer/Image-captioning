import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, d_model: int):
        """Encoder class to extract feature maps from images

        Args:
            `d_model` (`int`): size of text embedding
        """
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        # [B, 2048, H, W]
        resnet_output_channels = 2048
        self.resnet = nn.Sequential(*modules).eval()
        self.resnet.requires_grad_(False)
        self.linear = nn.Linear(
            in_features=resnet_output_channels,
            out_features=d_model
        )

        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.resnet(images)
        # [B, feature_maps, H, W] -> [B, feature_maps, H*W] -> [B, H*W, feature_maps]
        features = features.flatten(start_dim=-2, end_dim=-1).movedim(-1, 1)
        # [B, H*W, feature_maps] -> [B, H*W, d_model]
        features = self.linear(features)
        return features