from ..net_modules.model import Model
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from torch.utils.data import DataLoader
import torchvision.transforms as T
from skimage.io import imread
import torch
import numpy as np
pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE_TO = (224, 224)

class History:
    def __init__(self, history_path: str) -> None:
        """Aux class to evaluate training results

        Args:
            `history_path` (`str`): path to logs from training, for example `./lightning_logs/version_0/metrics.csv`
        """
        self.df = pd.read_csv(history_path)
        self.df.drop(columns=['test_CE', 'step', 'epoch'], inplace=True)
        df = {col: self.df[col].dropna().tolist() for col in self.df.columns}
        self.df = pd.DataFrame(df)
        
    def draw_history(self):
        """Draw graphs for train loss, val loss and learning rate"""
        epochs = self.df.index.to_series() + 1
        # Metrics figure
        metrics_fig = go.Figure()
        layout_config = dict(
            font=dict(size=16),
            width=1200,
            height=600,
            title_x=0.5
        )
        # Train loss
        metrics_fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.df['train_CE'],
                name='<b>train_loss</b>',
                marker=dict(color='#00BFFF')
            )
        )
        # Val loss
        metrics_fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.df['val_CE'],
                name='<b>val_loss</b>',
                marker=dict(color='#DC143C')
            )
        )
        metrics_fig.update_xaxes(title='<b>Epoch</b>')
        metrics_fig.update_yaxes(title='<b>Cross entropy loss</b>')
        metrics_fig.update_layout(title='<b>Loss history</b>', **layout_config)
        # Learning rate figure
        lr_fig = go.Figure()
        lr_fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.df['lr-Adam'],
                marker=dict(color='#a653ec')
            )
        )
        lr_fig.update_xaxes(title='<b>Epoch</b>')
        lr_fig.update_yaxes(title='<b>Learning rate</b>')
        lr_fig.update_layout(
            title='<b>Learning rate history</b>', 
            **layout_config,
            yaxis=dict(exponentformat='power')
        )
        # Show and save
        metrics_fig.write_image('imgs/history/metrics.png')
        metrics_fig.show()
        lr_fig.write_image('imgs/history/lr.png')
        lr_fig.show()
        
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
            orig_img = self.inv_normalizer(processed_img).cpu().permute(1, 2, 0).numpy()
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
        if orig_img.shape[-1] > 3:
            orig_img = orig_img[..., :-1]
        elif orig_img.shape[-1] < 3:
            raise ValueError(f'Image must have 3 (RGB) or 4 (RGBA) channels, got {orig_img.shape[-1]} channels')
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
        