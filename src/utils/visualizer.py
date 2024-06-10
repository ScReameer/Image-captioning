from ..model import Model
from ..data_processing.vocabulary import Vocabulary
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

class HistoryVisualizer:
    def __init__(self, history_path: str) -> None:
        self.df = pd.read_csv(history_path)
        self.df.drop(columns=['train_CE_step', 'test_CE', 'step', 'epoch'], inplace=True)
        self.df.rename(columns={'train_CE_epoch': 'train_CE'}, inplace=True)
        df = {col: self.df[col].dropna().tolist() for col in self.df.columns}
        self.df = pd.DataFrame(df)
        
    def visualize_history(self):
        epochs = self.df.index.to_series() + 1
        # Metrics figure
        metrics_fig = go.Figure()
        layout_config = dict(
            font=dict(size=18),
            width=900,
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
        # Show plots
        metrics_fig.write_image('imgs/history/metrics.png')
        metrics_fig.show()
        lr_fig.write_image('imgs/history/lr.png')
        lr_fig.show()
        
class PredictionsVisualizer:
    def __init__(self, vocab: Vocabulary) -> None:
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
        self.vocab = vocab
    
    def caption_dataloader(self, dataloader: DataLoader, model: Model, n_samples=5):
        iter_loader = iter(dataloader)
        for _ in range(n_samples):
            img, _ = next(iter_loader)
            processed_img = img[0] # [C, H, W]
            orig_img = self.inv_normalizer(processed_img).cpu().permute(1, 2, 0).numpy()
            self._show_img_with_caption(processed_img, orig_img, model)
            
    def caption_single_image(self, path_or_url: str, model: Model):
        orig_img = imread(path_or_url) # [H, W, C]
        if orig_img.shape[-1] > 3:
            orig_img = orig_img[..., :-1]
        processed_img = self.transforms(orig_img) # [C, H, W]
        self._show_img_with_caption(processed_img, orig_img, model)
        
    def _show_img_with_caption(self, processed_img: torch.Tensor, orig_img: np.ndarray, model: Model):
        prediction = ' '.join([self.vocab.idx2word[idx] for idx in model.predict(processed_img.to(DEVICE))][1:-1])
        print(prediction)
        px.imshow(orig_img).update_xaxes(visible=False).update_yaxes(visible=False).show()
            