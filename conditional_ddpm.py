from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from tqdm import tqdm

from Unets import (
    ConditionalUnet_baseline,
    ConditionalUnet_baseline_64,
    ConditionalUnet_baseline_attn,
    ConditionalUnet_baseline_attn_64,
    ConditionalUnet_MinAdaGN,
    ConditionalUnet_MinAdaGN_64,
)
from training_log import TrainingLossLogger
from utils import ddpm_schedules


def load_GameScene(data_root="./multigames_datasets", batch_size=128, img_size: int = 28):
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = ImageFolder(root=data_root, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(
        f"[Dataset] 共 {len(dataset)} 张图, {img_size}×{img_size}, "
        f"类别映射: {dataset.class_to_idx}"
    )
    return dataloader


class ConditionalDDPM(nn.Module):
    def __init__(self, eps_model, betas, T, device):
        super(ConditionalDDPM, self).__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device

    def forward(self, x, y):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)
        eps = torch.randn_like(x)
        x_t = (
            torch.sqrt(self.alpha_bar[t, None, None, None]) * x
            + torch.sqrt(1 - self.alpha_bar[t, None, None, None]) * eps
        )

        return self.mse_loss(eps, self.eps_model(x_t, y, t / self.T))

    def sample(self, n_sample, y, size):
        x_t = torch.randn(n_sample, *size).to(self.device)

        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            eps = self.eps_model(x_t, y, t_is)
            x_t = (
                1
                / torch.sqrt(self.alpha[t])
                * (
                    x_t
                    - eps
                    * (1 - self.alpha[t])
                    / torch.sqrt(1 - self.alpha_bar[t])
                )
                + self.sigma[t] * z
            )

        return x_t


def train_conditional_diffusion(
    diffusion,
    device,
    n_epoch=100,
    sample_dir="log/samples_conditional_baseline",
    file_tag="baseline",
    *,
    img_size: int = 28,
    data_root: str = "./multigames_datasets",
    arch: str = "baseline",
):
    diffusion.to(device)

    dataloader = load_GameScene(data_root=data_root, img_size=img_size)
    optim = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    loss_logger = TrainingLossLogger(
        os.path.join(sample_dir, "training_loss.json"),
        scheme=f"conditional_{arch}",
        paradigm="conditional",
        backbone=arch,
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
        with torch.no_grad():
            y = (torch.arange(0, 12) % 3).to(device)
            xh = diffusion.sample(12, y, (3, img_size, img_size))
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=3)
            save_image(grid, f"{sample_dir}/ddpm_sample_{file_tag}_{i}.png")

        if i == last_epoch:
            # 文件名含总 epoch，避免与 50/100 等不同训练长度混淆（最后一轮索引仍为 last_epoch）
            ckpt_name = f"{sample_dir}/ddpm_game_{file_tag}_epochs{n_epoch}.pth"
            torch.save(diffusion.state_dict(), ckpt_name)
            print(f"[Checkpoint] 已保存最终权重: {ckpt_name}")


def build_eps_model(arch: str, img_size: int = 28):
    arch = arch.lower().strip()
    if img_size not in (28, 64):
        raise ValueError("img_size 仅支持 28 或 64")

    if arch == "baseline":
        if img_size == 64:
            return ConditionalUnet_baseline_64(in_channels=3, n_feat=256, n_classes=3)
        return ConditionalUnet_baseline(in_channels=3, n_feat=256, n_classes=3)
    if arch == "baseline_attn":
        if img_size == 64:
            return ConditionalUnet_baseline_attn_64(
                in_channels=3, n_feat=256, n_classes=3
            )
        return ConditionalUnet_baseline_attn(in_channels=3, n_feat=256, n_classes=3)
    if arch == "min_adagn":
        if img_size == 64:
            return ConditionalUnet_MinAdaGN_64(in_channels=3, n_feat=256, n_classes=3)
        return ConditionalUnet_MinAdaGN(in_channels=3, n_feat=256, n_classes=3)
    raise ValueError(f"未知 arch={arch!r}，可选：baseline | baseline_attn | min_adagn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional DDPM（GameScene）")
    parser.add_argument(
        "--arch",
        type=str,
        default="baseline",
        choices=("baseline", "baseline_attn", "min_adagn"),
        help="baseline / baseline_attn（一处 self-attn）/ min_adagn",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=None,
        help="采样预览与 checkpoint 目录；默认含 arch、分辨率、_e<总epoch> 后缀",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--img_size",
        type=int,
        default=28,
        choices=(28, 64),
        help="28 或 64",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./multigames_datasets",
        help="ImageFolder 根目录（含各类子文件夹）",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    device = torch.device(device_type)

    sample_dir = args.sample_dir
    if sample_dir is None:
        sub = {
            "baseline": "samples_conditional_baseline",
            "baseline_attn": "samples_conditional_baseline_attn",
            "min_adagn": "samples_conditional_MinAdaGN",
        }
        suf = "_64" if args.img_size == 64 else ""
        sample_dir = "log/" + sub[args.arch] + suf + f"_e{args.epochs}"

    file_tags = {
        "baseline": "baseline",
        "baseline_attn": "baseline_attn",
        "min_adagn": "MinAdaGN",
    }
    file_tag = file_tags[args.arch]

    c_ddpm = ConditionalDDPM(
        eps_model=build_eps_model(args.arch, img_size=args.img_size),
        betas=(1e-4, 0.02),
        T=1000,
        device=device,
    )

    train_conditional_diffusion(
        diffusion=c_ddpm,
        device=device,
        n_epoch=args.epochs,
        sample_dir=sample_dir,
        file_tag=file_tag,
        img_size=args.img_size,
        data_root=args.data_root,
        arch=args.arch,
    )
