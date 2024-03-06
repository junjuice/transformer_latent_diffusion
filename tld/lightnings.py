import lightning as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import numpy as np
import safetensors
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
 
from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator
from tld.effnet import EfficientNetEncoder
from tld.previewer import Previewer
from tld.data import setup_data_2
import tld.danbooru as db
from train import ModelConfig

class DenoiserPL(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.denoiser = Denoiser(
            in_dim=config.n_channels,
            noise_embed_dims=config.noise_embed_dims,
            cond_dim=config.cond_dim,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
            n_layers=config.n_layers
        )

        self.effnet = EfficientNetEncoder()
        effnet_checkpoint = {}
        with safetensors.safe_open(config.eff_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                effnet_checkpoint[key] = f.get_tensor(key)
        self.effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
        self.effnet.eval().requires_grad_(False)
        del effnet_checkpoint

        self.previewer = Previewer()
        previewer_checkpoint = {}
        with safetensors.safe_open(config.prev_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                previewer_checkpoint[key] = f.get_tensor(key)
        self.previewer.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
        self.previewer.eval().requires_grad_(False)
        del previewer_checkpoint

        self.drop = nn.Dropout1d(0.15)
        self.save_hyperparameters()

    def random_noise(self, x):
        noise_level = torch.tensor(np.random.beta(self.config.beta_a, self.config.beta_b, len(x)), device=self.device)
        signal_level = 1 - noise_level
        noise = torch.randn_like(x)

        x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x

        x_noisy = x_noisy.to(self.device, dtype=self.dtype)
        noise_level = noise_level.to(self.device, dtype=self.dtype)
        return x_noisy, noise_level
    
    def forward(self, x, t, c):
        pred = self.denoiser.forward(x, t, c)
        return pred
    
    def loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduce="mean")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.denoiser.parameters(), lr=self.config.lr)
        return optimizer

    def train_dataloader(self):
        db.setup()
        length = 6_500_000 // self.trainer.world_size // self.config.batch_size
        chunk_size = 1128 // self.trainer.world_size
        chunk = range(chunk_size*self.trainer.global_rank, chunk_size*(self.trainer.global_rank+1))
        webdataset_paths = [self.config.webdataset_path.format(str(i).rjust(4, "0")) for i in chunk]
        #webdataset_paths = "file:F:/crawl2/data-0000.tar"
        dataloader = setup_data_2(
            bsz=self.config.batch_size,
            img_size=self.config.original_size,
            dataset_path=webdataset_paths,
            worker_limit=self.config.worker_limit,
            length=length
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        x, c = batch["images"], batch["embeddings"]
        c = self.drop(c)
        x_latent = self.effnet(x)
        x_noisy, noise_level = self.random_noise(x_latent)
        pred = self.forward(x_noisy, noise_level.view(-1,1), c)
        loss = self.loss_fn(pred, x_latent)
        self.log("train_loss", loss, prog_bar=True)
        return loss