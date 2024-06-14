from ..net.model import Model

import torch
import torchvision.transforms as T
import numpy as np
import plotly.io as pio
import plotly.express as px
from torch.utils.data import DataLoader
from skimage.io import imread

pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ImageNet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# ResNet input image size
RESIZE_TO = (224, 224)

class Predictor:
    def __init__(self) -> None:
        """Aux class to draw images with predicted caption"""
        self.inv_normalizer = T.Normalize(
            mean=[-m/s for m, s in zip(MEAN, STD)],
            std=[1/s for s in STD]
        )
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(RESIZE_TO),
            T.Normalize(
                mean=MEAN,
                std=STD
            )
        ])
    
    def caption_dataloader(self, dataloader: DataLoader, model: Model, n_samples=5) -> None:
        """Draw n samples from given dataloader with predicted caption

        Args:
            `dataloader` (`DataLoader`): dataloader with pairs (image, target_caption)
            `model` (`Model`): model to predict caption
            `n_samples` (`int`, optional): number of samples to draw with predicted caption. Defaults to `5`.
        """
        iter_loader = iter(dataloader)
        for _ in range(n_samples):
            img, _ = next(iter_loader)
            processed_img = img[0] # [C, H, W]
            orig_img = self.inv_normalizer(processed_img).cpu().permute(1, 2, 0).numpy() # [H, W, C]
            self._show_img_with_caption(processed_img, orig_img, model)
            
    def caption_single_image(self, path_or_url: str, model: Model) -> None:
        """Draw image from path or URL with predicted caption

        Args:
            `path_or_url` (`str`): path or URL to image. Image must have 3 (RGB) or 4 (RGBA) channels, otherwise raises `ValueError`
            `model` (`Model`): model to predict caption

        Raises:
            `ValueError`: image channels < 3
        """
        orig_img = imread(path_or_url) # [H, W, C]
        if 4 < orig_img.shape[-1] < 3:
            raise ValueError(f'Image must have 3 (RGB) or 4 (RGBA) channels, got {orig_img.shape[-1]} channels')
        elif orig_img.shape[-1] == 4:
            orig_img = orig_img[..., :-1] # RGBA -> RGB
        processed_img = self.transforms(orig_img) # [C, H, W]
        self._show_img_with_caption(processed_img, orig_img, model)
        
    def _show_img_with_caption(self, processed_img: torch.Tensor, orig_img: np.ndarray, model: Model) -> None:
        """Aux func to draw image and print caption

        Args:
            `processed_img` (`torch.Tensor`): transformed image of shape `[C, H, W]`
            `orig_img` (`np.ndarray`): original image of shape `[H, W, C]`
            `model` (`Model`): model to predict caption
        """
        prediction = model.predict(processed_img.to(DEVICE))
        print(prediction)
        px.imshow(orig_img).update_xaxes(visible=False).update_yaxes(visible=False).show()
        