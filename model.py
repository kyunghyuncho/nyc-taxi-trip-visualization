import torch
import torch.nn as nn
import pytorch_lightning as pl
import streamlit as st
from optimizer import Muon

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_layers: str, lr: float, optimizer_name: str, nonlinearity_name: str, input_noise_factor: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.input_noise_factor = input_noise_factor
        
        # Parse hidden layers
        try:
            hidden_dims = [int(x.strip()) for x in hidden_layers.split(',')]
        except ValueError:
             # Fallback if parsing fails
             hidden_dims = [64, 32]

        # Select nonlinearity
        if nonlinearity_name == 'ReLU':
            act_fn = nn.ReLU()
        elif nonlinearity_name == 'Tanh':
            act_fn = nn.Tanh()
        elif nonlinearity_name == 'GELU':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()

        # Build Encoder
        encoder_layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_d, h_dim))
            encoder_layers.append(act_fn)
            in_d = h_dim
        
        # Bottleneck strictly 2D
        encoder_layers.append(nn.Linear(in_d, 2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder
        decoder_layers = []
        in_d = 2
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_d, h_dim))
            decoder_layers.append(act_fn)
            in_d = h_dim
        
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def get_embeddings(self, x):
        """Extract the 2D bottleneck representations."""
        with torch.no_grad():
            self.eval()
            return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] # TensorDataset yields a tuple
        
        # Inject noise for denoising capability
        if self.input_noise_factor > 0:
            x_noisy = x + self.input_noise_factor * torch.randn_like(x)
            # Clip if necessary, though StandardScaler data is mean 0 std 1, so it shouldn't strictly require it
            x_hat = self(x_noisy)
        else:
            x_hat = self(x)
            
        loss = self.criterion(x_hat, x) # Reconstruction loss against ORIGINAL input
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        val_loss = self.criterion(x_hat, x)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_name == 'Muon':
            # Use custom Muon optimizer
            return Muon(self.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr)


class StreamlitProgressCallback(pl.Callback):
    """
    Updates the Streamlit progress bar and status text per epoch.
    """
    def __init__(self, progress_bar, status_text, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        
        # metric names depend on whether validation is run (usually run_epoch_end handles both)
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss = trainer.callback_metrics.get('val_loss')
        
        train_val_str = f"{train_loss.item():.4f}" if train_loss is not None else "N/A"
        val_val_str = f"{val_loss.item():.4f}" if val_loss is not None else "N/A"
        
        # Update progress bar
        progress = epoch / self.total_epochs
        self.progress_bar.progress(progress)
        
        # Update text
        self.status_text.text(f"Epoch {epoch}/{self.total_epochs} - Train Loss: {train_val_str} | Val Loss: {val_val_str}")
