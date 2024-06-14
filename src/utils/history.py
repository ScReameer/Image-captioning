import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'


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
            title_x=0.5,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
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
        if not os.path.exists('imgs/history'):
            os.makedirs('imgs/history')
        metrics_fig.write_image('imgs/history/metrics.png')
        metrics_fig.show()
        lr_fig.write_image('imgs/history/lr.png')
        lr_fig.show()