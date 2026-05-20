#!/usr/bin/env python3
"""
GameScene：训练 → 按类导出生成图 → 每类 FID → 追加 CSV。

生成图统一放在「一个总文件夹」下，再按分辨率与 epoch 分子文件夹，最后按方案分子文件夹：
  gen_fid_all/h64_e100/conditional_baseline/
  gen_fid_all/h64_e100/cfg_min_adagn/
  ...

一键跑满「默认 7 方案 × 两种分辨率」（极耗时）：
  cd /path/to/GameScene   # 含 conditional_ddpm.py 等
  python run_experiment_suite.py --data_root ./multigames_datasets --both_res --device cuda:0

只跑 64×64、默认 7 方案：
  python run_experiment_suite.py --data_root ./multigames_datasets --img_size 64

已有 ckpt、只导出 + FID：
  python run_experiment_suite.py --both_res --skip_train --data_root ./multigames_datasets

仅根据已有 CSV + gen_fid_all 生成论文表/图（FID 表、loss 曲线、定性生成图）：
  python run_experiment_suite.py --skip_train --skip_sample --paper_only --img_size 64

默认 7 种（baseline 三种范式 + MinAdaGN 三种 + attention 条件）：
  conditional_baseline, cfg_baseline, guided_baseline,
  conditional_min_adagn, cfg_min_adagn, guided_min_adagn,
  conditional_baseline_attn

兼容简写：baseline / min_adagn / baseline_attn / cfg / guided（cfg、guided 默认 backbone=baseline）。
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fid_utils import fid_per_class, list_class_names
from gamescene_schemes import (
    ckpt_path_cfg,
    ckpt_path_conditional,
    ckpt_paths_guided,
    parse_experiment_scheme,
)
from paper_assets import generate_paper_assets

DEFAULT_SCHEMES = (
    "conditional_baseline,cfg_baseline,guided_baseline,"
    "conditional_min_adagn,cfg_min_adagn,guided_min_adagn,"
    "conditional_baseline_attn"
)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败 ({r.returncode}): {' '.join(cmd)}")


def train_scheme(
    scheme_label: str,
    img_size: int,
    epochs: int,
    data_root: str,
    cwd: Path,
) -> None:
    py = sys.executable
    paradigm, backbone = parse_experiment_scheme(scheme_label)
    common = [
        "--img_size",
        str(img_size),
        "--epochs",
        str(epochs),
        "--data_root",
        data_root,
    ]
    if paradigm == "conditional":
        run_cmd([py, "conditional_ddpm.py", "--arch", backbone] + common, cwd=cwd)
    elif paradigm == "cfg":
        run_cmd(
            [py, "classifier_free_ddpm.py", "--backbone", backbone] + common,
            cwd=cwd,
        )
    elif paradigm == "guided":
        run_cmd(
            [py, "guided_ddpm.py", "--mode", "train_all", "--backbone", backbone]
            + common,
            cwd=cwd,
        )
    else:
        raise ValueError(paradigm)


def sample_scheme(
    scheme_label: str,
    img_size: int,
    epochs: int,
    data_root: str,
    out_dir: Path,
    device: str,
    cwd: Path,
    guided_grad_scale: float = 1.0,
) -> None:
    py = sys.executable
    paradigm, backbone = parse_experiment_scheme(scheme_label)
    base = [
        py,
        "sample_for_fid.py",
        "--scheme",
        scheme_label,
        "--img_size",
        str(img_size),
        "--match_data_root",
        data_root,
        "--out_dir",
        str(out_dir),
        "--device",
        device,
    ]
    if paradigm == "guided":
        eps, clf = ckpt_paths_guided(backbone, img_size, epochs)
        base += [
            "--ckpt_eps",
            eps,
            "--ckpt_classifier",
            clf,
            "--grad_scale",
            str(guided_grad_scale),
        ]
    elif paradigm == "cfg":
        base += ["--ckpt", ckpt_path_cfg(backbone, img_size, epochs)]
    else:
        base += ["--ckpt", ckpt_path_conditional(backbone, img_size, epochs)]
    run_cmd(base, cwd=cwd)


def resolve_img_sizes(args: argparse.Namespace) -> list[int]:
    if getattr(args, "img_sizes", None):
        out = []
        for x in args.img_sizes.split(","):
            x = x.strip()
            if not x:
                continue
            v = int(x)
            if v not in (28, 64):
                raise SystemExit(f"img_sizes 仅支持 28 或 64，收到 {v}")
            out.append(v)
        if not out:
            raise SystemExit("img_sizes 为空")
        return out
    if args.both_res:
        return [28, 64]
    return [args.img_size]


def main() -> None:
    p = argparse.ArgumentParser(
        description="训练 + 导出 + 每类 FID；生成图集中在 gen_root 下按 h×w / epoch / 方案分层"
    )
    p.add_argument("--cwd", type=str, default=".", help="工程根目录（含各 .py）")
    p.add_argument("--data_root", type=str, default="./multigames_datasets")
    p.add_argument("--img_size", type=int, default=64, choices=(28, 64), help="单分辨率时使用")
    p.add_argument(
        "--img_sizes",
        type=str,
        default=None,
        help="逗号分隔，如 28,64；若指定则覆盖 --img_size（与 --both_res 互斥优先本项）",
    )
    p.add_argument(
        "--both_res",
        action="store_true",
        help="同时跑 28 与 64（等价于 img_sizes=28,64）",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--schemes",
        type=str,
        default=DEFAULT_SCHEMES,
        help="逗号分隔；默认 7 项：baseline 三范式 + MinAdaGN 三范式 + conditional_baseline_attn（详见脚本顶部说明）",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fid_batch", type=int, default=64)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sample", action="store_true")
    p.add_argument("--out_csv", type=str, default="experiment_fid_per_class.csv")
    p.add_argument(
        "--gen_root",
        type=str,
        default="gen_fid_all",
        help="所有生成 FID 图片的根目录（其下为 h{size}_e{epochs}/{scheme}/）",
    )
    p.add_argument(
        "--guided_grad_scale",
        type=float,
        default=1.0,
        help="guided 采样时 classifier 梯度缩放（sample_for_fid --grad_scale）",
    )
    p.add_argument(
        "--paper_dir",
        type=str,
        default="paper_outputs",
        help="论文用表/图输出根目录（其下 h{size}_e{epochs}/tables|figures）",
    )
    p.add_argument(
        "--skip_paper_assets",
        action="store_true",
        help="不生成 paper_outputs 下的 FID 表 / loss 图 / 定性图",
    )
    p.add_argument(
        "--paper_only",
        action="store_true",
        help="仅根据已有 CSV 与 gen_fid_all 生成论文表/图（跳过训练、采样、FID）",
    )
    p.add_argument(
        "--paper_img_size",
        type=int,
        default=None,
        help="论文资产使用的分辨率（默认与 --img_size 相同；both_res 时默认 64）",
    )
    p.add_argument("--paper_samples_per_col", type=int, default=4)
    p.add_argument("--paper_paradigm_class", type=str, default="starpilot")
    args = p.parse_args()

    if args.img_sizes and args.both_res:
        print("[WARN] 同时指定 --img_sizes 与 --both_res，以 --img_sizes 为准", flush=True)

    cwd = Path(args.cwd).resolve()
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    img_sizes = resolve_img_sizes(args)

    gen_root = cwd / args.gen_root
    gen_root.mkdir(parents=True, exist_ok=True)

    csv_path = cwd / args.out_csv

    if args.paper_only:
        paper_img = args.paper_img_size or (64 if args.both_res else args.img_size)
        assets = generate_paper_assets(
            cwd=cwd,
            csv_path=csv_path,
            gen_root=gen_root,
            paper_dir=cwd / args.paper_dir,
            data_root=Path(args.data_root),
            img_size=paper_img,
            epochs=args.epochs,
            guided_grad_scale=args.guided_grad_scale,
            samples_per_col=args.paper_samples_per_col,
            paradigm_class=args.paper_paradigm_class,
        )
        print("[DONE] paper-only assets:", flush=True)
        for k, v in assets.items():
            print(f"  {k}: {v}", flush=True)
        return

    write_header = not csv_path.is_file()
    timestamp = datetime.now().isoformat(timespec="seconds")

    fieldnames = [
        "timestamp",
        "scheme",
        "img_size",
        "epochs",
        "fid_device",
        "gen_subdir",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        class_names = list_class_names(args.data_root)
        for name in class_names:
            fieldnames.append(f"fid_{name}")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for img_size in img_sizes:
            layer = gen_root / f"h{img_size}_e{args.epochs}"
            layer.mkdir(parents=True, exist_ok=True)

            for scheme in schemes:
                out_dir = layer / scheme
                rel = out_dir.relative_to(cwd)
                if not args.skip_train:
                    train_scheme(scheme, img_size, args.epochs, args.data_root, cwd)
                if not args.skip_sample:
                    sample_scheme(
                        scheme,
                        img_size,
                        args.epochs,
                        args.data_root,
                        out_dir,
                        args.device,
                        cwd,
                        guided_grad_scale=args.guided_grad_scale,
                    )
                fids = fid_per_class(
                    args.data_root,
                    out_dir,
                    device=args.device,
                    batch_size=args.fid_batch,
                )
                row = {
                    "timestamp": timestamp,
                    "scheme": scheme,
                    "img_size": img_size,
                    "epochs": args.epochs,
                    "fid_device": args.device,
                    "gen_subdir": str(rel).replace("\\", "/"),
                }
                for name in class_names:
                    row[f"fid_{name}"] = f"{fids[name]:.6f}"
                writer.writerow(row)
                f.flush()
                print(f"[FID] {scheme} h{img_size}: {fids}", flush=True)

    print(f"[DONE] 生成图根目录: {gen_root}")
    print(f"[DONE] CSV: {csv_path}")

    if not args.skip_paper_assets:
        paper_img = args.paper_img_size
        if paper_img is None:
            paper_img = 64 if (len(img_sizes) > 1 or args.both_res) else img_sizes[0]
        try:
            assets = generate_paper_assets(
                cwd=cwd,
                csv_path=csv_path,
                gen_root=gen_root,
                paper_dir=cwd / args.paper_dir,
                data_root=Path(args.data_root),
                img_size=paper_img,
                epochs=args.epochs,
                guided_grad_scale=args.guided_grad_scale,
                samples_per_col=args.paper_samples_per_col,
                paradigm_class=args.paper_paradigm_class,
            )
            print(f"[DONE] 论文表/图: {cwd / args.paper_dir / f'h{paper_img}_e{args.epochs}'}", flush=True)
            for k, v in assets.items():
                print(f"  {k}: {v}", flush=True)
        except Exception as e:
            print(f"[WARN] 论文资产生成失败: {e}", flush=True)


if __name__ == "__main__":
    main()
