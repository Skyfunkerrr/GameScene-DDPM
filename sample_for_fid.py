#!/usr/bin/env python3
"""
导出按「类别子文件夹」组织的生成图，供 fid_utils.fid_per_class 使用。

--scheme 推荐写法：
  conditional_baseline | conditional_min_adagn | conditional_baseline_attn
  cfg_baseline | cfg_min_adagn
  guided_baseline | guided_min_adagn

简写（与旧版兼容）：baseline / min_adagn / baseline_attn / cfg / guided
（cfg、guided 表示 backbone=baseline 的 CFG / guided）。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from classifier_free_ddpm import ClassifierFreeDDPM, build_eps_for_cfg
from conditional_ddpm import ConditionalDDPM, build_eps_model
from fid_utils import min_per_class_counts
from gamescene_schemes import parse_experiment_scheme
from guided_ddpm import GuidedDDPM, UnconditionalDDPM, build_classifier, build_uncond_eps


def class_label_map(data_root: str | Path, img_size: int) -> tuple[list[str], dict[str, int]]:
    """与训练时 ImageFolder 完全一致的类别名与下标（避免手写 enumerate 与数据集不一致）。"""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    ds = ImageFolder(str(data_root), transform=tf)
    return list(ds.classes), dict(ds.class_to_idx)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_conditional_ddpm(device: torch.device, arch: str, img_size: int) -> ConditionalDDPM:
    eps = build_eps_model(arch, img_size=img_size)
    return ConditionalDDPM(eps, betas=(1e-4, 0.02), T=1000, device=device)


def load_classifier_free(device: torch.device, img_size: int, backbone: str) -> ClassifierFreeDDPM:
    return ClassifierFreeDDPM(
        build_eps_for_cfg(backbone, img_size),
        betas=(1e-4, 0.02),
        T=1000,
        device=device,
        drop_prob=0.1,
    )


@torch.no_grad()
def export_per_class_subdirs(
    sample_fn,
    out_root: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    n_per_class: int,
    channels: int,
    hw: int,
    max_batch: int,
) -> None:
    """sample_fn(bs, y_tensor, size_tuple) -> tensor [-1,1]"""
    out_root.mkdir(parents=True, exist_ok=True)
    size_t = (channels, hw, hw)

    for name in class_names:
        c = class_to_idx[name]
        sub = out_root / name
        sub.mkdir(parents=True, exist_ok=True)
        left = n_per_class
        pbar = tqdm(total=n_per_class, desc=f"{name}(y={c})")
        idx = 0
        while left > 0:
            bs = min(max_batch, left)
            y = torch.full((bs,), c, dtype=torch.long, device=sample_fn.device)
            xh = sample_fn(bs, y, size_t)
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            for i in range(bs):
                save_image(xh[i].cpu(), sub / f"{idx:06d}_gen.png")
                idx += 1
            left -= bs
            pbar.update(bs)
        pbar.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export class-wise PNG folders for per-class FID.")
    p.add_argument("--ckpt", type=str, default=None, help="ConditionalDDPM / ClassifierFree 整网 ckpt")
    p.add_argument("--ckpt_eps", type=str, default=None, help="guided：无条件 DDPM 整网 ckpt")
    p.add_argument("--ckpt_classifier", type=str, default=None, help="guided：分类器 ckpt")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--match_data_root", type=str, required=True, help="真实数据 ImageFolder 根（含各类子文件夹）")
    p.add_argument("--n_classes", type=int, default=3)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--size", type=int, default=None, help="默认等于 --img_size")
    p.add_argument("--max_batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scheme",
        type=str,
        default="conditional_baseline",
        help="如 conditional_baseline, cfg_min_adagn, guided_baseline；简写 baseline/cfg/guided 等见 gamescene_schemes",
    )
    p.add_argument("--img_size", type=int, default=28, choices=(28, 64))
    p.add_argument("--cfg_w", type=float, default=1.0, help="cfg 采样 guidance 权重")
    p.add_argument("--grad_scale", type=float, default=1.0, help="guided 分类器梯度缩放")
    return p.parse_args()


class _CondSampler:
    def __init__(self, diffusion: ConditionalDDPM):
        self.diffusion = diffusion
        self.device = diffusion.device

    def __call__(self, bs, y, size_t):
        return self.diffusion.sample(bs, y, size_t)


class _CfgSampler:
    def __init__(self, m: ClassifierFreeDDPM, guide_w: float):
        self.m = m
        self.guide_w = guide_w
        self.device = m.device

    def __call__(self, bs, y, size_t):
        return self.m.sample(bs, y, size_t, guide_w=self.guide_w)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    hw = args.img_size if args.size is None else args.size
    if hw != args.img_size:
        print("错误：--size 须等于 --img_size", file=sys.stderr)
        sys.exit(2)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    names, n_per = min_per_class_counts(args.match_data_root)
    folder_classes, class_to_idx = class_label_map(args.match_data_root, args.img_size)
    print(f"[INFO] ImageFolder 类别映射: {class_to_idx}", flush=True)
    if set(names) != set(folder_classes):
        print(
            f"[WARN] fid_utils 子文件夹列表 {names} 与 ImageFolder.classes {folder_classes} 不一致",
            file=sys.stderr,
        )
    if len(names) != args.n_classes:
        print(
            f"[WARN] 子文件夹数 {len(names)} 与 n_classes={args.n_classes} 不一致，按实际名称导出",
            file=sys.stderr,
        )

    try:
        paradigm, backbone = parse_experiment_scheme(args.scheme)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    if paradigm == "conditional":
        diffusion = build_conditional_ddpm(device, backbone, args.img_size)
        if not args.ckpt:
            print("错误：conditional 须指定 --ckpt", file=sys.stderr)
            sys.exit(2)
        diffusion.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
        diffusion.eval()
        diffusion.to(device)
        sampler = _CondSampler(diffusion)
    elif paradigm == "cfg":
        if not args.ckpt:
            print("错误：cfg 须指定 --ckpt", file=sys.stderr)
            sys.exit(2)
        if backbone not in ("baseline", "min_adagn"):
            print("错误：CFG 仅支持 backbone baseline | min_adagn", file=sys.stderr)
            sys.exit(2)
        m = load_classifier_free(device, args.img_size, backbone)
        m.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
        m.eval()
        m.to(device)
        sampler = _CfgSampler(m, args.cfg_w)
    elif paradigm == "guided":
        if not args.ckpt_eps or not args.ckpt_classifier:
            print("错误：guided 须指定 --ckpt_eps 与 --ckpt_classifier", file=sys.stderr)
            sys.exit(2)
        if backbone not in ("baseline", "min_adagn"):
            print("错误：guided 仅支持 backbone baseline | min_adagn", file=sys.stderr)
            sys.exit(2)
        ud = UnconditionalDDPM(
            build_uncond_eps(args.img_size, backbone),
            betas=(1e-4, 0.02),
            T=1000,
            device=device,
        )
        ud.load_state_dict(torch.load(args.ckpt_eps, map_location=device), strict=True)
        ud.eval()
        ud.to(device)
        clf = build_classifier(args.img_size)
        clf.load_state_dict(torch.load(args.ckpt_classifier, map_location=device), strict=True)
        clf.eval()
        clf.to(device)
        g_ddpm = GuidedDDPM(
            ud.eps_model,
            betas=(1e-4, 0.02),
            T=1000,
            device=device,
            classifier=clf,
            grad_scale=args.grad_scale,
        )
        print(
            f"[INFO] guided ckpt_eps={args.ckpt_eps}\n"
            f"       ckpt_classifier={args.ckpt_classifier}\n"
            f"       grad_scale={args.grad_scale}",
            flush=True,
        )
        print(f"[INFO] 每类生成 {n_per} 张 → {args.out_dir}")
        export_per_class_subdirs_guided(
            g=g_ddpm,
            classifier=clf,
            out_root=Path(args.out_dir),
            class_names=names,
            class_to_idx=class_to_idx,
            n_per_class=n_per,
            channels=args.channels,
            hw=hw,
            max_batch=args.max_batch,
        )
        print(f"[DONE] {args.out_dir}")
        return
    else:
        print(f"错误：未知范式 {paradigm!r}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] 每类生成 {n_per} 张 → {args.out_dir}")
    export_per_class_subdirs(
        sample_fn=sampler,
        out_root=Path(args.out_dir),
        class_names=names,
        class_to_idx=class_to_idx,
        n_per_class=n_per,
        channels=args.channels,
        hw=hw,
        max_batch=args.max_batch,
    )
    print(f"[DONE] {args.out_dir}")


def export_per_class_subdirs_guided(
    g: GuidedDDPM,
    classifier: torch.nn.Module,
    out_root: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    n_per_class: int,
    channels: int,
    hw: int,
    max_batch: int,
) -> None:
    """guided 采样不可整体 no_grad（需分类器反传）。"""
    out_root.mkdir(parents=True, exist_ok=True)
    size_t = (channels, hw, hw)
    device = g.device
    idx_to_name = {v: k for k, v in class_to_idx.items()}

    for name in class_names:
        c = class_to_idx[name]
        sub = out_root / name
        sub.mkdir(parents=True, exist_ok=True)
        left = n_per_class
        pbar = tqdm(total=n_per_class, desc=f"{name}(y={c})")
        idx = 0
        first_batch_checked = False
        while left > 0:
            bs = min(max_batch, left)
            y = torch.full((bs,), c, dtype=torch.long, device=device)
            xh = g.sample(bs, y, size_t)
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            if not first_batch_checked:
                with torch.no_grad():
                    pred = classifier((xh * 2 - 1).to(device)).argmax(1).cpu().tolist()
                hit = sum(1 for p in pred if p == c)
                print(
                    f"[CHECK] 目标 {name} (y={c}) 首 batch 分类器 argmax 命中 {hit}/{bs}；"
                    f"预测分布: {[idx_to_name.get(p, str(p)) for p in pred]}",
                    flush=True,
                )
                if hit == 0:
                    print(
                        f"[WARN] 生成图完全未被噪声分类器判为目标类，"
                        f"易出现「文件夹名是 {name} 但画面像别的游戏」；"
                        f"请检查 ckpt 是否成对、或降低 --grad_scale（当前 {g._scale}）",
                        file=sys.stderr,
                    )
                first_batch_checked = True
            with torch.no_grad():
                for i in range(bs):
                    save_image(xh[i].cpu(), sub / f"{idx:06d}_gen.png")
                    idx += 1
            left -= bs
            pbar.update(bs)
        pbar.close()


if __name__ == "__main__":
    main()
