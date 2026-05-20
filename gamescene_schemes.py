"""GameScene 实验 scheme 解析与 checkpoint 路径（conditional / CFG / guided × backbone）。"""
from __future__ import annotations

from pathlib import Path


def parse_experiment_scheme(raw: str) -> tuple[str, str]:
    """
    返回 (paradigm, backbone)。
    paradigm: conditional | cfg | guided
    backbone: baseline | min_adagn | baseline_attn（仅 conditional 可用 baseline_attn）
    """
    s = raw.strip().lower().replace("-", "_")
    legacy = {
        "baseline": ("conditional", "baseline"),
        "min_adagn": ("conditional", "min_adagn"),
        "baseline_attn": ("conditional", "baseline_attn"),
        "cfg": ("cfg", "baseline"),
        "guided": ("guided", "baseline"),
    }
    if s in legacy:
        return legacy[s]

    if s.startswith("conditional_"):
        b = s[len("conditional_") :]
        if b not in ("baseline", "min_adagn", "baseline_attn"):
            raise ValueError(
                f"未知 conditional backbone {b!r}，应为 baseline | min_adagn | baseline_attn"
            )
        return ("conditional", b)

    if s.startswith("cfg_"):
        b = s[len("cfg_") :]
        if b not in ("baseline", "min_adagn"):
            raise ValueError(f"未知 cfg backbone {b!r}，应为 baseline | min_adagn")
        return ("cfg", b)

    if s.startswith("guided_"):
        b = s[len("guided_") :]
        if b not in ("baseline", "min_adagn"):
            raise ValueError(f"未知 guided backbone {b!r}，应为 baseline | min_adagn")
        return ("guided", b)

    raise ValueError(
        f"未知 scheme {raw!r}。示例：conditional_baseline, cfg_min_adagn, guided_baseline"
    )


def ckpt_path_conditional(arch: str, img_size: int, epochs: int) -> str:
    suf = "_64" if img_size == 64 else ""
    sub = {
        "baseline": "samples_conditional_baseline",
        "min_adagn": "samples_conditional_MinAdaGN",
        "baseline_attn": "samples_conditional_baseline_attn",
    }
    tags = {"baseline": "baseline", "min_adagn": "MinAdaGN", "baseline_attn": "baseline_attn"}
    if arch not in sub:
        raise ValueError(arch)
    return f"log/{sub[arch]}{suf}_e{epochs}/ddpm_game_{tags[arch]}_epochs{epochs}.pth"


def ckpt_path_cfg(backbone: str, img_size: int, epochs: int) -> str:
    suf = "_64" if img_size == 64 else ""
    return (
        f"log/samples_classifier_free_{backbone}{suf}_e{epochs}/"
        f"ddpm_game_cfg_epochs{epochs}.pth"
    )


def ckpt_paths_guided(backbone: str, img_size: int, epochs: int) -> tuple[str, str]:
    suf = "_64" if img_size == 64 else ""
    base_u = f"log/guided_unconditional_{backbone}{suf}_e{epochs}"
    base_c = f"log/guided_classifier_{backbone}{suf}_e{epochs}"
    return (
        f"{base_u}/ddpm_game_uncond_epochs{epochs}.pth",
        f"{base_c}/classifier_guided_epochs{epochs}.pth",
    )


def training_log_dirs_for_scheme(scheme: str, img_size: int, epochs: int) -> list[str]:
    """返回该 scheme 训练时写入 training_loss*.json 的目录（guided 为两个）。"""
    paradigm, backbone = parse_experiment_scheme(scheme)
    suf = "_64" if img_size == 64 else ""
    ep = epochs
    if paradigm == "conditional":
        sub = {
            "baseline": "samples_conditional_baseline",
            "min_adagn": "samples_conditional_MinAdaGN",
            "baseline_attn": "samples_conditional_baseline_attn",
        }
        return [f"log/{sub[backbone]}{suf}_e{ep}"]
    if paradigm == "cfg":
        return [f"log/samples_classifier_free_{backbone}{suf}_e{ep}"]
    if paradigm == "guided":
        u, c = ckpt_paths_guided(backbone, img_size, ep)
        return [str(Path(u).parent), str(Path(c).parent)]
    raise ValueError(paradigm)


# 论文主表 / 主图默认方案（64×64）
PAPER_MAIN_SCHEMES_64 = [
    "conditional_baseline",
    "cfg_baseline",
    "conditional_min_adagn",
    "cfg_min_adagn",
    "conditional_baseline_attn",
]

PAPER_PARADIGM_SCHEMES_64 = [
    "conditional_baseline",
    "cfg_baseline",
    "guided_baseline",
]

PAPER_ABLATION_SCHEMES_64 = [
    "conditional_baseline",
    "conditional_min_adagn",
    "conditional_baseline_attn",
    "cfg_min_adagn",
]
