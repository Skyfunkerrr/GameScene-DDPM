#!/usr/bin/env python3
"""
从 experiment_fid_per_class.csv、gen_fid_all/、log/*/training_loss.json
生成论文用表格（FID 数值）与插图（训练 loss 曲线、定性生成图对比）。

输出目录默认 paper_outputs/h{size}_e{epochs}/：
  tables/   — CSV + Markdown（FID 主表、消融表、实验设置表）
  figures/  — PNG（loss 曲线、主定性对比、范式对比）
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from fid_utils import list_class_names
from gamescene_schemes import (
    PAPER_ABLATION_SCHEMES_64,
    PAPER_MAIN_SCHEMES_64,
    PAPER_PARADIGM_SCHEMES_64,
    parse_experiment_scheme,
    training_log_dirs_for_scheme,
)

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _read_fid_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"FID CSV 不存在: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _filter_rows(
    rows: list[dict[str, str]],
    *,
    img_size: int,
    epochs: int,
    schemes: list[str] | None = None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    want = set(schemes) if schemes else None
    for r in rows:
        if int(r["img_size"]) != img_size or int(r["epochs"]) != epochs:
            continue
        if want is not None and r["scheme"] not in want:
            continue
        out.append(r)
    order = {s: i for i, s in enumerate(schemes or [])}
    if schemes:
        out.sort(key=lambda r: order.get(r["scheme"], 999))
    return out


def _fid_cols(class_names: list[str]) -> list[str]:
    return [f"fid_{n}" for n in class_names]


def _row_mean_fid(row: dict[str, str], class_names: list[str]) -> float:
    vals = [float(row[f"fid_{n}"]) for n in class_names]
    return sum(vals) / len(vals)


def write_fid_table(
    rows: list[dict[str, str]],
    class_names: list[str],
    out_csv: Path,
    out_md: Path,
    title: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scheme", *_fid_cols(class_names), "fid_mean"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "scheme": r["scheme"],
                    **{f"fid_{n}": f"{float(r[f'fid_{n}']):.2f}" for n in class_names},
                    "fid_mean": f"{_row_mean_fid(r, class_names):.2f}",
                }
            )

    header = "| Scheme | " + " | ".join(class_names) + " | Mean |"
    sep = "|---|" + "|".join(["---:"] * (len(class_names) + 1)) + "|"
    lines = [f"## {title}", "", header, sep]
    for r in rows:
        cells = [r["scheme"]] + [f"{float(r[f'fid_{n}']):.2f}" for n in class_names]
        cells.append(f"{_row_mean_fid(r, class_names):.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_settings_table(
    out_md: Path,
    *,
    data_root: str,
    img_size: int,
    epochs: int,
    class_names: list[str],
    class_counts: dict[str, int] | None,
    guided_grad_scale: float,
    cfg_drop_prob: float,
    cfg_guide_w: float,
) -> None:
    lines = [
        "## Experimental settings",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset root | `{data_root}` |",
        f"| Classes | {', '.join(class_names)} |",
        f"| Train resolution | {img_size}×{img_size} |",
        f"| Epochs | {epochs} |",
        f"| Diffusion steps T | 1000 |",
        f"| β schedule | linear 1e-4 → 0.02 |",
        f"| Optimizer | Adam, lr=2e-4 |",
        f"| Batch size | 128 |",
        f"| Metric | Per-class FID (Inception) |",
        f"| CFG train drop_prob | {cfg_drop_prob} |",
        f"| CFG sample guide_w | {cfg_guide_w} |",
        f"| Guided grad_scale | {guided_grad_scale} |",
    ]
    if class_counts:
        for n in class_names:
            lines.append(f"| Images in `{n}/` | {class_counts[n]} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_loss_json(path: Path) -> list[tuple[int, float]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pts: list[tuple[int, float]] = []
    for row in data.get("epochs", []):
        if "loss_ema" in row:
            pts.append((int(row["epoch"]), float(row["loss_ema"])))
        elif "loss" in row:
            pts.append((int(row["epoch"]), float(row["loss"])))
    return pts


def plot_loss_curves(
    cwd: Path,
    schemes: list[str],
    img_size: int,
    epochs: int,
    out_png: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 未安装 matplotlib，跳过 loss 曲线", flush=True)
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_line = False
    for scheme in schemes:
        paradigm, _ = parse_experiment_scheme(scheme)
        for log_dir in training_log_dirs_for_scheme(scheme, img_size, epochs):
            log_path = cwd / log_dir / "training_loss.json"
            pts = _load_loss_json(log_path)
            if not pts:
                continue
            xs, ys = zip(*pts)
            label = scheme if paradigm != "guided" else f"{scheme} ({log_dir.split('/')[-1]})"
            ax.plot(xs, ys, label=label, linewidth=1.2)
            any_line = True

    if not any_line:
        plt.close(fig)
        print("[WARN] 无 training_loss.json，跳过 fig_loss_curves", flush=True)
        return False

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (EMA)")
    ax.set_title(f"Training loss ({img_size}×{img_size})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def _list_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(folder.glob(ext))
    return sorted(paths, key=lambda p: p.name)


def _pick_images(folder: Path, n: int, seed: int) -> list[Path]:
    paths = _list_images(folder)
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(paths)), n))
    return [paths[i] for i in idx]


def _load_rgb(path: Path, size: int):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def _make_labeled_grid(
    columns: list[tuple[str, list[Path]]],
    *,
    img_size: int,
    cell_px: int,
    out_png: Path,
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 未安装 matplotlib，跳过定性图", flush=True)
        return False

    nrows = max((len(p) for _, p in columns), default=0)
    if nrows == 0:
        print(f"[WARN] 无可用图片，跳过 {out_png.name}", flush=True)
        return False

    ncols = len(columns)
    fig_w = max(6, 1.8 * ncols)
    fig_h = max(3, 1.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(title, fontsize=11)

    for j, (col_title, paths) in enumerate(columns):
        for i in range(nrows):
            ax = axes[i, j]
            ax.axis("off")
            if i == 0:
                ax.set_title(col_title, fontsize=9)
            if i < len(paths):
                ax.imshow(_load_rgb(paths[i], img_size), interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build_qualitative_main_figure(
    gen_layer: Path,
    real_root: Path,
    class_names: list[str],
    schemes: list[str],
    *,
    img_size: int,
    samples_per_col: int,
    seed: int,
    out_png: Path,
) -> bool:
    columns: list[tuple[str, list[Path]]] = []
    for cls in class_names:
        real_dir = real_root / cls
        columns.append((f"Real\n{cls}", _pick_images(real_dir, samples_per_col, seed + hash(cls) % 997)))
    for scheme in schemes:
        for cls in class_names:
            gen_dir = gen_layer / scheme / cls
            columns.append(
                (
                    f"{scheme}\n{cls}",
                    _pick_images(gen_dir, samples_per_col, seed + hash(scheme + cls) % 997),
                )
            )
    return _make_labeled_grid(
        columns,
        img_size=img_size,
        cell_px=img_size,
        out_png=out_png,
        title=f"Qualitative comparison ({img_size}×{img_size}, {samples_per_col} samples per cell)",
    )


def build_qualitative_by_class_rows(
    gen_layer: Path,
    real_root: Path,
    class_names: list[str],
    schemes: list[str],
    *,
    img_size: int,
    samples_per_col: int,
    seed: int,
    out_png: Path,
) -> bool:
    """每行一个游戏：列 = Real | scheme1 | scheme2 | ..."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    col_labels = ["Real"] + schemes
    ncols = len(col_labels)
    nrows = len(class_names)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.0 * ncols, 2.0 * nrows),
        squeeze=False,
    )
    fig.suptitle(f"Per-game samples ({img_size}×{img_size})", fontsize=11)

    for ri, cls in enumerate(class_names):
        sources: list[tuple[str, Path | None]] = [("Real", real_root / cls)]
        for s in schemes:
            sources.append((s, gen_layer / s / cls))

        for ci, (label, folder) in enumerate(sources):
            ax = axes[ri, ci]
            ax.axis("off")
            if ri == 0:
                ax.set_title(label, fontsize=8)
            if ri == 0 and ci == 0:
                pass
            ax.set_ylabel(cls if ci == 0 else "", fontsize=8)
            if folder is None or not folder.is_dir():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            paths = _pick_images(folder, 1, seed + ri * 17 + ci)
            if paths:
                ax.imshow(_load_rgb(paths[0], img_size), interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build_paradigm_figure(
    gen_layer: Path,
    real_root: Path,
    target_class: str,
    schemes: list[str],
    *,
    img_size: int,
    samples_per_col: int,
    seed: int,
    out_png: Path,
) -> bool:
    columns: list[tuple[str, list[Path]]] = [
        ("Real", _pick_images(real_root / target_class, samples_per_col, seed)),
    ]
    for scheme in schemes:
        columns.append(
            (
                scheme,
                _pick_images(gen_layer / scheme / target_class, samples_per_col, seed + 1),
            )
        )
    return _make_labeled_grid(
        columns,
        img_size=img_size,
        cell_px=img_size,
        out_png=out_png,
        title=f"Conditioning paradigms on {target_class} ({img_size}×{img_size})",
    )


def _count_class_images(data_root: Path, class_names: list[str]) -> dict[str, int]:
    from fid_utils import count_images_flat

    return {n: count_images_flat(data_root / n) for n in class_names}


def generate_paper_assets(
    *,
    cwd: Path,
    csv_path: Path,
    gen_root: Path,
    paper_dir: Path,
    data_root: Path,
    img_size: int,
    epochs: int,
    class_names: list[str] | None = None,
    guided_grad_scale: float = 1.0,
    cfg_drop_prob: float = 0.1,
    cfg_guide_w: float = 1.0,
    samples_per_col: int = 4,
    paradigm_class: str = "starpilot",
    seed: int = 0,
) -> dict[str, Path]:
    cwd = cwd.resolve()
    class_names = class_names or list_class_names(data_root)
    layer = gen_root / f"h{img_size}_e{epochs}"
    out_base = paper_dir / f"h{img_size}_e{epochs}"
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows_all = _read_fid_csv(csv_path)
    written: dict[str, Path] = {}

    # 表：实验设置
    counts = _count_class_images(Path(data_root), class_names)
    settings_md = tables_dir / "table_experiment_settings.md"
    write_settings_table(
        settings_md,
        data_root=str(data_root),
        img_size=img_size,
        epochs=epochs,
        class_names=class_names,
        class_counts=counts,
        guided_grad_scale=guided_grad_scale,
        cfg_drop_prob=cfg_drop_prob,
        cfg_guide_w=cfg_guide_w,
    )
    written["settings"] = settings_md

    # 表：主 FID
    main_rows = _filter_rows(
        rows_all, img_size=img_size, epochs=epochs, schemes=PAPER_MAIN_SCHEMES_64
    )
    if main_rows:
        write_fid_table(
            main_rows,
            class_names,
            tables_dir / "table_fid_main.csv",
            tables_dir / "table_fid_main.md",
            f"Main FID results ({img_size}×{img_size})",
        )
        written["fid_main_csv"] = tables_dir / "table_fid_main.csv"

    # 表：消融 FID
    abl_rows = _filter_rows(
        rows_all, img_size=img_size, epochs=epochs, schemes=PAPER_ABLATION_SCHEMES_64
    )
    if abl_rows:
        write_fid_table(
            abl_rows,
            class_names,
            tables_dir / "table_fid_ablation.csv",
            tables_dir / "table_fid_ablation.md",
            f"Ablation FID ({img_size}×{img_size})",
        )
        written["fid_ablation_csv"] = tables_dir / "table_fid_ablation.csv"

    # 表：范式对比（含 guided）
    para_rows = _filter_rows(
        rows_all, img_size=img_size, epochs=epochs, schemes=PAPER_PARADIGM_SCHEMES_64
    )
    if para_rows:
        write_fid_table(
            para_rows,
            class_names,
            tables_dir / "table_fid_paradigm.csv",
            tables_dir / "table_fid_paradigm.md",
            f"Paradigm comparison FID ({img_size}×{img_size})",
        )
        written["fid_paradigm_csv"] = tables_dir / "table_fid_paradigm.csv"

    # 附录：全表
    all_rows = _filter_rows(rows_all, img_size=img_size, epochs=epochs, schemes=None)
    if all_rows:
        write_fid_table(
            all_rows,
            class_names,
            tables_dir / "table_fid_all_schemes.csv",
            tables_dir / "table_fid_all_schemes.md",
            f"All schemes FID ({img_size}×{img_size})",
        )

    # 图：loss 曲线
    loss_schemes = [
        "conditional_baseline",
        "cfg_min_adagn",
        "conditional_baseline_attn",
    ]
    loss_png = figures_dir / "fig_loss_curves.png"
    if plot_loss_curves(cwd, loss_schemes, img_size, epochs, loss_png):
        written["loss_fig"] = loss_png

    # 图：按行每类游戏（论文主定性图，推荐）
    qual_png = figures_dir / "fig_qualitative_per_game.png"
    main_schemes = [s for s in PAPER_MAIN_SCHEMES_64 if (layer / s).is_dir()]
    if main_schemes and build_qualitative_by_class_rows(
        layer,
        Path(data_root),
        class_names,
        main_schemes,
        img_size=img_size,
        samples_per_col=1,
        seed=seed,
        out_png=qual_png,
    ):
        written["qual_per_game"] = qual_png

    # 图：范式对比（starpilot）
    if paradigm_class in class_names:
        para_png = figures_dir / f"fig_paradigm_{paradigm_class}.png"
        para_schemes = [s for s in PAPER_PARADIGM_SCHEMES_64 if (layer / s).is_dir()]
        if para_schemes and build_paradigm_figure(
            layer,
            Path(data_root),
            paradigm_class,
            para_schemes,
            img_size=img_size,
            samples_per_col=min(3, samples_per_col),
            seed=seed + 2,
            out_png=para_png,
        ):
            written["paradigm_fig"] = para_png

    readme = out_base / "README.md"
    lines = [
        f"# Paper assets (h{img_size}, e{epochs})",
        "",
        "## Tables (FID numbers)",
        "- `tables/table_experiment_settings.md` — hyperparameters",
        "- `tables/table_fid_main.csv` — main results",
        "- `tables/table_fid_ablation.csv` — ablation",
        "- `tables/table_fid_paradigm.csv` — conditional / CFG / guided",
        "",
        "## Figures",
        "- `figures/fig_loss_curves.png` — **training loss** (EMA per epoch)",
        "- `figures/fig_qualitative_per_game.png` — **generated samples** vs real (one row per class)",
        f"- `figures/fig_paradigm_{paradigm_class}.png` — **generated samples** paradigm comparison",
        "",
        "Generated images are read from:",
        f"`{layer}/<scheme>/<class>/*.png`",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    written["readme"] = readme
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper tables (FID) and figures (loss / samples)")
    p.add_argument("--cwd", type=str, default=".")
    p.add_argument("--csv", type=str, default="experiment_fid_per_class.csv")
    p.add_argument("--gen_root", type=str, default="gen_fid_all")
    p.add_argument("--paper_dir", type=str, default="paper_outputs")
    p.add_argument("--data_root", type=str, default="./multigames_datasets")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--samples_per_col", type=int, default=4)
    p.add_argument("--paradigm_class", type=str, default="starpilot")
    p.add_argument("--guided_grad_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = generate_paper_assets(
        cwd=Path(args.cwd),
        csv_path=Path(args.cwd) / args.csv,
        gen_root=Path(args.cwd) / args.gen_root,
        paper_dir=Path(args.cwd) / args.paper_dir,
        data_root=Path(args.data_root),
        img_size=args.img_size,
        epochs=args.epochs,
        guided_grad_scale=args.guided_grad_scale,
        samples_per_col=args.samples_per_col,
        paradigm_class=args.paradigm_class,
        seed=args.seed,
    )
    print("[DONE] paper assets:", flush=True)
    for k, v in paths.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
