import os, pdb
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

import torchvision.transforms as T
from PIL import Image

from nerf.provider import rand_poses
from torch_efficient_distloss import eff_distloss
from kornia.losses import ssim_loss, inverse_depth_smoothness_loss, total_variation
from kornia.filters import gaussian_blur2d

# Visualization function (unchanged)
def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)
    mi = np.min(x)
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)
    return x_

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

# Optimized get_rays function with memory efficiency
@torch.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                          torch.linspace(0, H - 1, H, device=device), indexing='ij')
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    if N > 0:
        inds = torch.randint(0, H * W, size=[N], device=device)
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)

    rays_o = poses[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)
    return {'rays_o': rays_o, 'rays_d': rays_d}

# Trainer class optimized
class Trainer:
    def __init__(self, name, model, opt, guidance, device=None):
        self.name = name
        self.model = model.to(device)
        self.opt = opt
        self.guidance = guidance
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        self.console = Console()

        self.global_step = 0
        self.epoch = 0
        self.fp16 = opt.fp16
        self.ckpt_path = f"checkpoints/{name}"

        # Initialize EMA
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=opt.ema_decay)

    def train_step(self, data):
        """
        Perform a single training step.
        """
        rays_o, rays_d = data['rays_o'], data['rays_d']
        B, N = rays_o.shape[:2]

        # Optional gradient checkpointing
        outputs = checkpoint(self.model.render, rays_o, rays_d, self.device)

        pred_rgb = outputs['image'].view(B, N, -1)
        loss = F.mse_loss(pred_rgb, data['gt_rgb'])

        # Optimization step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.ema:
            self.ema.update()

        return loss.item()

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_loader):
            try:
                with torch.cuda.amp.autocast(self.fp16):
                    loss = self.train_step(batch)
                epoch_loss += loss
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("[WARN] CUDA OOM. Skipping batch.")
                    torch.cuda.empty_cache()
                else:
                    raise e
        return epoch_loss / len(train_loader)

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            loss = self.train_epoch(train_loader)
            print(f"Loss: {loss:.4f}")
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        os.makedirs(self.ckpt_path, exist_ok=True)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   os.path.join(self.ckpt_path, f"checkpoint_{epoch}.pth"))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

# Example usage
# Trainer setup and usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel()  # Replace with actual model
    opt = Options()  # Replace with actual options/config
    guidance = GuidanceModule()  # Replace with actual guidance module

    trainer = Trainer("my_experiment", model, opt, guidance, device=device)

    train_loader = DataLoader(MyDataset(), batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    trainer.train(train_loader, num_epochs=opt.epochs)
