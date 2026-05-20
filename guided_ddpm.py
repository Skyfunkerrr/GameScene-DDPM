"""
Classifier Guidance（GameScene）：无条件 DDPM（baseline 或 MinAdaGN 风格 ε）+ 噪声分类器。
--backbone baseline | min_adagn
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from conditional_ddpm import load_GameScene
from Unets import (
    EncoderUnet,
    EncoderUnet_64,
    UnconditionalMinAdaGN,
    UnconditionalMinAdaGN_64,
    UnconditionalUnet_64,
    Unet,
)
from training_log import TrainingLossLogger
from utils import ddpm_schedules


def classifier_grad_fn(x, classifier, y, scale: float = 1.0):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        grad = torch.autograd.grad(selected.sum(), x_in)[0] * scale
        return grad


class UnconditionalDDPM(nn.Module):
    def __init__(self, eps_model, betas, T, device):
        super().__init__()
        self.eps_model = eps_model
        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)
        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device

    def forward(self, x):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)
        eps = torch.randn_like(x)
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(
            1 - self.alpha_bar[t, None, None, None]
        ) * eps
        t_ratio = (t.float() / self.T).view(-1, 1, 1, 1)
        return self.mse_loss(eps, self.eps_model(x_t, t_ratio))

    def sample(self, n_sample, size):
        x_t = torch.randn(n_sample, *size).to(self.device)
        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device).repeat(n_sample, 1, 1, 1)
            eps = self.eps_model(x_t, t_is)
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


class GuidedDDPM(nn.Module):
    def __init__(
        self,
        eps_model,
        betas,
        T,
        device,
        classifier,
        grad_scale: float = 1.0,
    ):
        super().__init__()
        self.eps_model = eps_model
        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)
        self.T = T
        self.device = device
        self.classifier = classifier
        self._scale = grad_scale

    def sample(self, n_sample, y, size):
        x_t = torch.randn(n_sample, *size).to(self.device)
        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device).repeat(n_sample, 1, 1, 1)
            with torch.no_grad():
                eps = self.eps_model(x_t, t_is)
            x_t_mean = (
                1
                / torch.sqrt(self.alpha[t])
                * (
                    x_t
                    - eps * (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])
                )
            )
            x_t_variance = self.sigma[t]
            gradient = classifier_grad_fn(x_t_mean, self.classifier, y, self._scale)
            x_t_mean = x_t_mean + x_t_variance * gradient
            x_t = x_t_mean + self.sigma[t] * z
        return x_t


def train_unconditional(
    diffusion: UnconditionalDDPM,
    device: torch.device,
    n_epoch: int,
    sample_dir: str,
    img_size: int,
    data_root: str,
    backbone: str = "baseline",
):
    diffusion.to(device)
    optim = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
    loader = load_GameScene(data_root=data_root, img_size=img_size)
    os.makedirs(sample_dir, exist_ok=True)
    loss_logger = TrainingLossLogger(
        f"{sample_dir}/training_loss.json",
        scheme=f"guided_{backbone}",
        paradigm="guided",
        backbone=backbone,
        img_size=img_size,
        phase="unconditional_eps",
    )
    last = n_epoch - 1
    tag = "uncond"

    for i in range(n_epoch):
        diffusion.train()
        loss_ema = None
        pbar = tqdm(loader)
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = diffusion(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        loss_logger.log_epoch(i, loss_ema)

        diffusion.eval()
        with torch.no_grad():
            xh = diffusion.sample(12, (3, img_size, img_size))
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            save_image(make_grid(xh, nrow=4), f"{sample_dir}/ddpm_sample_{tag}_{i}.png")

        if i == last:
            path = f"{sample_dir}/ddpm_game_{tag}_epochs{n_epoch}.pth"
            torch.save(diffusion.state_dict(), path)
            print(f"[Uncond checkpoint] {path}")


def train_classifier_guidance_branch(
    classifier: nn.Module,
    diffusion: UnconditionalDDPM,
    device: torch.device,
    n_epoch: int,
    out_dir: str,
    data_root: str,
    img_size: int,
    backbone: str = "baseline",
):
    classifier.to(device)
    diffusion.to(device)
    optim = torch.optim.Adam(classifier.parameters(), lr=2e-4)
    ce = nn.CrossEntropyLoss()
    loader = load_GameScene(data_root=data_root, img_size=img_size)
    os.makedirs(out_dir, exist_ok=True)
    loss_logger = TrainingLossLogger(
        f"{out_dir}/training_loss.json",
        scheme=f"guided_{backbone}",
        paradigm="guided",
        backbone=backbone,
        img_size=img_size,
        phase="noise_classifier",
    )
    last = n_epoch - 1
    epoch_loss_ema = None
    epoch_acc_ema = None

    for i in range(n_epoch):
        classifier.train()
        pbar = tqdm(loader)
        batch_losses: list[float] = []
        batch_accs: list[float] = []
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(1, diffusion.T, (x.shape[0],)).to(device)
            eps = torch.randn_like(x)
            x_t = torch.sqrt(diffusion.alpha_bar[t, None, None, None]) * x + torch.sqrt(
                1 - diffusion.alpha_bar[t, None, None, None]
            ) * eps
            logits = classifier(x_t)
            loss = ce(logits, y)
            loss.backward()
            acc = (logits.argmax(1) == y).float().mean()
            batch_losses.append(loss.item())
            batch_accs.append(acc.item())
            pbar.set_description(f"loss: {loss.item():.4f}, acc: {acc:.3f}")
            optim.step()

        if batch_losses:
            mean_loss = sum(batch_losses) / len(batch_losses)
            mean_acc = sum(batch_accs) / len(batch_accs)
            if epoch_loss_ema is None:
                epoch_loss_ema = mean_loss
                epoch_acc_ema = mean_acc
            else:
                epoch_loss_ema = 0.9 * epoch_loss_ema + 0.1 * mean_loss
                epoch_acc_ema = 0.9 * epoch_acc_ema + 0.1 * mean_acc
            loss_logger.log_epoch(
                i,
                epoch_loss_ema,
                cls_acc_ema=epoch_acc_ema,
            )

        if i == last:
            path = f"{out_dir}/classifier_guided_epochs{n_epoch}.pth"
            torch.save(classifier.state_dict(), path)
            print(f"[Classifier checkpoint] {path}")


def build_uncond_eps(img_size: int, backbone: str = "baseline") -> nn.Module:
    b = backbone.lower().strip()
    if b == "min_adagn":
        if img_size == 64:
            return UnconditionalMinAdaGN_64(in_channels=3, n_feat=256)
        return UnconditionalMinAdaGN(in_channels=3, n_feat=256)
    if b == "baseline":
        if img_size == 64:
            return UnconditionalUnet_64(in_channels=3, n_feat=256)
        return Unet(in_channels=3, n_feat=256)
    raise ValueError(f"--backbone 仅支持 baseline | min_adagn，收到 {backbone!r}")


def build_classifier(img_size: int) -> nn.Module:
    if img_size == 64:
        return EncoderUnet_64(in_channels=3, n_feat=256, n_classes=3)
    return EncoderUnet(in_channels=3, n_feat=256, n_classes=3)


def run_train_all(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    suf = "_64" if args.img_size == 64 else ""
    ep = args.epochs
    uncond_dir = args.uncond_dir or f"log/guided_unconditional_{args.backbone}{suf}_e{ep}"

    eps = build_uncond_eps(args.img_size, args.backbone)
    uncond = UnconditionalDDPM(eps, betas=(1e-4, 0.02), T=1000, device=device).to(device)
    train_unconditional(
        uncond,
        device,
        n_epoch=args.epochs,
        sample_dir=uncond_dir,
        img_size=args.img_size,
        data_root=args.data_root,
        backbone=args.backbone,
    )

    ckpt = f"{uncond_dir}/ddpm_game_uncond_epochs{args.epochs}.pth"
    state = torch.load(ckpt, map_location=device)
    uncond.load_state_dict(state)
    uncond.eval()

    clf_dir = args.classifier_dir or f"log/guided_classifier_{args.backbone}{suf}_e{ep}"
    clf = build_classifier(args.img_size)
    train_classifier_guidance_branch(
        clf,
        uncond,
        device,
        n_epoch=args.epochs,
        out_dir=clf_dir,
        data_root=args.data_root,
        img_size=args.img_size,
        backbone=args.backbone,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier-guided DDPM（GameScene）")
    parser.add_argument("--mode", type=str, default="train_all", choices=("train_all",))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=28, choices=(28, 64))
    parser.add_argument("--data_root", type=str, default="./multigames_datasets")
    parser.add_argument("--backbone", type=str, default="baseline", choices=("baseline", "min_adagn"))
    parser.add_argument("--uncond_dir", type=str, default=None)
    parser.add_argument("--classifier_dir", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train_all":
        run_train_all(args)
