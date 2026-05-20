"""
Classifier-Free Diffusion Guidance（GameScene）。
ε 网络可选 baseline 或 MinAdaGN（--backbone）；训练随机丢弃类别；采样 CFG 组合。
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from conditional_ddpm import load_GameScene
from training_log import TrainingLossLogger
from Unets import (
    ConditionalUnet_baseline,
    ConditionalUnet_baseline_64,
    ConditionalUnet_MinAdaGN,
    ConditionalUnet_MinAdaGN_64,
)
from utils import ddpm_schedules


class ClassifierFreeDDPM(nn.Module):
    def __init__(self, eps_model, betas, T, device, drop_prob=0.1):
        super().__init__()
        self.eps_model = eps_model
        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)
        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device
        self.drop_prob = drop_prob

    def forward(self, x, y):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)
        eps = torch.randn_like(x)
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(
            1 - self.alpha_bar[t, None, None, None]
        ) * eps
        context_mask = torch.bernoulli(torch.zeros_like(y, dtype=torch.float) + self.drop_prob).to(
            self.device
        )
        return self.mse_loss(eps, self.eps_model(x_t, y, t / self.T, context_mask))

    def sample(self, n_sample, y, size, guide_w=0.0):
        x_t = torch.randn(n_sample, *size).to(self.device)
        y_double = y.repeat(2)
        context_mask = torch.zeros_like(y_double).to(self.device)
        context_mask[n_sample:] = 1.0

        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device).repeat(n_sample, 1, 1, 1)
            x_t_double = x_t.repeat(2, 1, 1, 1)
            t_is_double = t_is.repeat(2, 1, 1, 1)
            eps = self.eps_model(x_t_double, y_double, t_is_double, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_t = (
                1
                / torch.sqrt(self.alpha[t])
                * (
                    x_t
                    - eps * (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])
                )
                + self.sigma[t] * z
            )
        return x_t


def train_classifier_free(
    diffusion: ClassifierFreeDDPM,
    device: torch.device,
    n_epoch: int,
    sample_dir: str,
    file_tag: str,
    img_size: int,
    data_root: str,
    backbone: str = "baseline",
):
    diffusion.to(device)
    optim = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
    dataloader = load_GameScene(data_root=data_root, img_size=img_size)
    os.makedirs(sample_dir, exist_ok=True)
    loss_logger = TrainingLossLogger(
        f"{sample_dir}/training_loss.json",
        scheme=f"cfg_{backbone}",
        paradigm="cfg",
        backbone=backbone,
        img_size=img_size,
    )
    last_epoch = n_epoch - 1

    for i in range(n_epoch):
        diffusion.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = diffusion(x, y)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        loss_logger.log_epoch(i, loss_ema)

        diffusion.eval()
        guide_w = 1.0
        with torch.no_grad():
            ys = (torch.arange(0, 12) % 3).to(device)
            xh = diffusion.sample(12, ys, (3, img_size, img_size), guide_w=guide_w)
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            save_image(make_grid(xh, nrow=3), f"{sample_dir}/ddpm_sample_{file_tag}_{i}.png")

        if i == last_epoch:
            path = f"{sample_dir}/ddpm_game_{file_tag}_epochs{n_epoch}.pth"
            torch.save(diffusion.state_dict(), path)
            print(f"[Checkpoint] {path}")


def build_eps_for_cfg(backbone: str, img_size: int):
    b = backbone.lower().strip()
    if b == "baseline":
        if img_size == 64:
            return ConditionalUnet_baseline_64(in_channels=3, n_feat=256, n_classes=3)
        return ConditionalUnet_baseline(in_channels=3, n_feat=256, n_classes=3)
    if b == "min_adagn":
        if img_size == 64:
            return ConditionalUnet_MinAdaGN_64(in_channels=3, n_feat=256, n_classes=3)
        return ConditionalUnet_MinAdaGN(in_channels=3, n_feat=256, n_classes=3)
    raise ValueError(f"--backbone 仅支持 baseline | min_adagn，收到 {backbone!r}")


def build_eps_baseline(img_size: int):
    """兼容旧调用。"""
    return build_eps_for_cfg("baseline", img_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classifier-Free DDPM（GameScene，baseline / MinAdaGN）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--img_size", type=int, default=28, choices=(28, 64))
    p.add_argument("--data_root", type=str, default="./multigames_datasets")
    p.add_argument("--sample_dir", type=str, default=None)
    p.add_argument("--drop_prob", type=float, default=0.1)
    p.add_argument("--backbone", type=str, default="baseline", choices=("baseline", "min_adagn"))
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    ep = args.epochs
    suf = "_64" if args.img_size == 64 else ""
    sample_dir = args.sample_dir or (
        f"log/samples_classifier_free_{args.backbone}{suf}_e{ep}"
    )
    tag = "cfg"

    model = ClassifierFreeDDPM(
        eps_model=build_eps_for_cfg(args.backbone, args.img_size),
        betas=(1e-4, 0.02),
        T=1000,
        device=device,
        drop_prob=args.drop_prob,
    )
    train_classifier_free(
        model,
        device,
        n_epoch=args.epochs,
        sample_dir=sample_dir,
        file_tag=tag,
        img_size=args.img_size,
        data_root=args.data_root,
        backbone=args.backbone,
    )
