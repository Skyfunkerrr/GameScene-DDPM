"""按游戏类别（ImageFolder 子文件夹名）分别计算 FID；不计算全局混合 FID。"""
from __future__ import annotations

from pathlib import Path


def list_class_names(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not names:
        raise FileNotFoundError(f"无子文件夹: {root}")
    return names


def count_images_flat(d: Path) -> int:
    n = 0
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        n += len(list(d.glob(ext)))
    return n


def min_per_class_counts(data_root: str | Path) -> tuple[list[str], int]:
    """每类真实图数量的最小值，用于生成对齐。"""
    names = list_class_names(data_root)
    counts = []
    for n in names:
        c = count_images_flat(Path(data_root) / n)
        if c == 0:
            raise FileNotFoundError(f"{data_root}/{n} 下无图像")
        counts.append(c)
    return names, min(counts)


def fid_per_class(
    real_root: str | Path,
    gen_root: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 64,
    dims: int = 2048,
    num_workers: int = 0,
) -> dict[str, float]:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    names = list_class_names(real_root)
    out: dict[str, float] = {}
    gen_root = Path(gen_root)
    for name in names:
        rp = Path(real_root) / name
        gp = gen_root / name
        if not gp.is_dir():
            raise FileNotFoundError(f"生成目录缺失: {gp}")
        fid = calculate_fid_given_paths(
            [str(rp), str(gp)],
            batch_size,
            device,
            dims,
            num_workers,
        )
        out[name] = float(fid)
    return out
