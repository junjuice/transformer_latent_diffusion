#!/usr/bin/env python3

import copy
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from PIL import Image
import wandb
import torchvision.utils as vutils

from accelerate import Accelerator
import torchvision

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import safetensors

from diffusers import AutoencoderKL

from tld.bucketeer import Bucketeer
from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator
from tld.effnet import EfficientNetEncoder
from tld.previewer import Previewer
from tld.data import setup_data
import tld.danbooru as db

import gc
import os

def eval_gen(diffuser: DiffusionGenerator, batch):
    class_guidance=4.5
    seed=10
    out, _ = diffuser.generate( 
        batch=batch,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
    )

    out: Image.Image = to_pil((vutils.make_grid((out+1)/2, nrow=8, padding=4)).float().clip(0, 1))

    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_layer(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

to_pil = torchvision.transforms.ToPILImage()

def update_ema(ema_model, model, alpha=0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1-alpha)


@dataclass
class ModelConfig:
    original_size: int = 768
    cond_dim: int = 512
    embed_dim: int = 512
    n_layers: int = 6
    clip_embed_size: int = 768
    scaling_factor: int = 8
    patch_size: int = 2
    image_size: int = 32 
    n_channels: int = 16
    dropout: float = 0
    mlp_multiplier: int = 4
    batch_size: int = 16
    class_guidance: int = 3
    lr: float = 3e-4
    n_epoch: int = 100
    alpha: float = 0.999
    noise_embed_dims: int = 128
    diffusion_n_iter: int = 35
    from_scratch: bool = True
    run_id: str = None
    model_name: str = None
    beta_a: float = 0.75
    beta_b: float = 0.75
    save_and_eval_every_iters: int = 1000
    eff_path: str = "models/effnet_encoder.safetensors"
    prev_path: str = "models/previewer.safetensors"
    webdataset_path: str = "https://huggingface.co/datasets/KBlueLeaf/danbooru2023-webp-2Mpixel/resolve/main/images/data-{}.tar"
    worker_limit: int = 0

def main(config: ModelConfig = ModelConfig()):
    """main train loop to be used with accelerate"""

    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")

    accelerator.print("Loading Data:")
    webdataset_paths = [config.webdataset_path.format(str(i).rjust(4, "0")) for i in range(1128)]
    train_loader = setup_data(config.batch_size, config.original_size, webdataset_paths, config.worker_limit)
    
    print("Loading EffnetEncoder...")
    effnet = EfficientNetEncoder()
    effnet_checkpoint = {}
    with safetensors.safe_open(config.eff_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            effnet_checkpoint[key] = f.get_tensor(key)
    effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
    effnet.eval().requires_grad_(False)
    del effnet_checkpoint

    print("Loading Previewer...")
    previewer = Previewer()
    previewer_checkpoint = {}
    with safetensors.safe_open(config.prev_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            previewer_checkpoint[key] = f.get_tensor(key)
    previewer.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
    previewer.eval().requires_grad_(False)
    del previewer_checkpoint

    if accelerator.is_main_process:
        effnet = effnet.to(accelerator.device)
        previewer = previewer.to(accelerator.device)
   
    print("Setup model and optimizer...")
    model = Denoiser(
        in_dim=config.n_channels, 
        image_size=config.image_size, 
        noise_embed_dims=config.noise_embed_dims,
        cond_dim=config.cond_dim, 
        patch_size=config.patch_size, 
        embed_dim=config.embed_dim, 
        dropout=config.dropout,
        n_layers=config.n_layers
    )
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    accelerator.print("Compiling model:")
    #model = torch.compile(model)

    if not config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(config.model_name, run_path=f"junjuice0/ntt_diffusion/runs/{config.run_id}",
                      replace=True)
        full_state_dict = torch.load(config.model_name)
        model.load_state_dict(full_state_dict['model_ema'])
        optimizer.load_state_dict(full_state_dict['opt_state'])
        global_step = full_state_dict['global_step']
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, effnet=effnet, previewer=previewer, device=accelerator.device, model_dtype=torch.bfloat16)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(
        model, train_loader, optimizer
    )
    train_loader = Bucketeer(train_loader, config.batch_size, density=config.image_size ** 2, factor=32, interpolate_nearest=False, length=6_500_000)

    accelerator.init_trackers(
        project_name="ntt_diffusion",
        config=asdict(config)
    )
    drop = nn.Dropout1d(0.15)
    accelerator.print("Training on:", accelerator.device)
    accelerator.print("Parameters:", count_parameters(model))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ### Train:
    for i in range(1, config.n_epoch+1):
        accelerator.print(f'epoch: {i}')         
        for _ in tqdm(range(train_loader.length//config.batch_size)):
            batch = next(train_loader)
            x = batch["images"].to(accelerator.device)
            y = db.get_conditions(batch).to(accelerator.device)

            x = diffuser.effnet(x)
            noise_level = torch.tensor(np.random.beta(config.beta_a, config.beta_b, len(x)), device=accelerator.device)
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()

            label = y
            label = drop(label)

            if global_step % config.save_and_eval_every_iters == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    with torch.autocast(accelerator.device.type, dtype=torch.bfloat16):
                        out = eval_gen(diffuser=diffuser, batch=batch)
                    out.save('step{}.jpg'.format(global_step))
                    accelerator.log({f"step: {global_step}": wandb.Image('step{}.jpg'.format(global_step))}, step=global_step)
                    
                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        'model_ema':ema_model.state_dict(),
                        'opt_state':opt_unwrapped.state_dict(),
                        'global_step':global_step
                    }
                    accelerator.save(full_state_dict, config.model_name)
                    wandb.save(config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                with torch.autocast(accelerator.device.type, dtype=torch.bfloat16):
                    pred: torch.Tensor = model(x_noisy, noise_level.view(-1,1), label)
                loss = loss_fn(pred, x)
                accelerator.log(
                    {
                    "train_loss":loss.item(), 
                    "pred_mean": pred.mean().item(), 
                    "pred_max": pred.max().item(), 
                    "pred_min": pred.min().item(), 
                    "step": global_step
                    },
                    step=global_step
                )
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=config.alpha)

            global_step += 1
    accelerator.end_training()
            
if __name__ == "__main__":
    main(ModelConfig())