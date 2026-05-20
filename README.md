# GameScene-DDPM

Diffusion experiments on **multi-game Procgen-style screenshots** (caveflyer, coinrun, starpilot). The repo supports:

1. **Dataset construction** from Procgen rollouts (GIF → per-class PNG folders)
2. **Three conditioning paradigms**: class-conditional DDPM, classifier-free guidance (CFG), classifier-guided DDPM
3. **Backbone variants**: baseline U-Net, MinAdaGN, optional self-attention (conditional only)
4. **Per-class FID** evaluation and optional **paper tables/figures** export

---

## Requirements

Core training & evaluation:

```bash
pip install torch torchvision tqdm numpy pillow matplotlib pytorch-fid
```

Dataset recording (Procgen + GIF I/O):

```bash
pip install procgen gym imageio
```

Or install from file:

```bash
pip install -r requirements.txt
```

GPU recommended for training and FID (`cuda:0` by default in scripts).

---

## Dataset layout

After preparation, training expects an **ImageFolder** root (default `./multigames_datasets`):

```text
multigames_datasets/
  caveflyer/    # label 0 (alphabetical)
  coinrun/      # label 1
  starpilot/    # label 2
```

Each class folder contains PNG frames extracted from rollouts.

---

## 1. Build the dataset (Procgen → PNGs)

### How record_procgen_rollout.py works
This is only an example aimed to explain how record_procgen_rollout.py works.
Full workflow is contained in the following collect.py.
[Procgen](https://github.com/openai/procgen) rollout to a GIF:

```bash
python record_procgen_rollout.py --env coinrun --steps 500 --out rollouts/coinrun.gif
```

Useful flags: `--env`, `--steps`, `--out`, `--start-level`, `--num-levels`, `--distribution-mode`, `--seed`.

### Step 1 — Batch record (all three games)

`collect_all.py` calls `record_procgen_rollout.py` for **coinrun, starpilot, caveflyer** and writes GIFs under `multigames_rollouts/`.

**Before running**, edit the `output_base` path at the top of `collect_all.py` (default in repo may point to an AutoDL path), e.g.:

```python
output_base = "./multigames_rollouts"
```

Then:

```bash
python collect_all.py
```

### Step 2 — Extract frames to class folders

`extract.py` splits each GIF into PNGs under `multigames_datasets/<game>/`.

**Before running**, set paths at the top of `extract.py`, e.g.:

```python
gif_dir = "./multigames_rollouts"
output_dir = "./multigames_datasets"
```

Then:

```bash
python extract.py
```

---

## 2. Train & evaluate (recommended: one entry script)

`run_experiment_suite.py` runs **train → per-class sampling → per-class FID → append CSV**, and optionally builds **paper assets** (tables + figures).

### Full pipeline (64×64, default 7 schemes)

```bash
python run_experiment_suite.py \
  --data_root ./multigames_datasets \
  --img_size 64 \
  --epochs 100 \
  --device cuda:0 \
  --guided_grad_scale 0.2
```

Default schemes (comma-separated internally):

| Scheme                      | Paradigm                 | Backbone             |
| --------------------------- | ------------------------ | -------------------- |
| `conditional_baseline`      | Class-conditional DDPM   | baseline U-Net       |
| `cfg_baseline`              | Classifier-free guidance | baseline             |
| `guided_baseline`           | Classifier-guided        | baseline             |
| `conditional_min_adagn`     | Class-conditional        | MinAdaGN             |
| `cfg_min_adagn`             | CFG                      | MinAdaGN             |
| `guided_min_adagn`          | Classifier-guided        | MinAdaGN             |
| `conditional_baseline_attn` | Class-conditional        | baseline + attention |

**Note:** Classifier-guided sampling is **sensitive** to `grad_scale` and per-class stability; treat it as a comparison baseline, not the main method.

### Custom scheme list (e.g. guided baseline only)

```bash
python run_experiment_suite.py \
  --data_root ./multigames_datasets \
  --img_size 64 \
  --schemes conditional_baseline,cfg_baseline,guided_baseline,conditional_min_adagn,cfg_min_adagn,conditional_baseline_attn \
  --guided_grad_scale 0.2
```

### Skip training or sampling

```bash
# Only sample + FID (checkpoints under log/)
python run_experiment_suite.py --skip_train --data_root ./multigames_datasets --img_size 64

# Only regenerate paper tables/figures from existing CSV + gen_fid_all/
python run_experiment_suite.py --skip_train --skip_sample --paper_only --img_size 64
```

### Outputs

| Path                                              | Content                                                      |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `log/`                                            | Checkpoints (`*.pth`), training previews, `training_loss.json` |
| `gen_fid_all/h{size}_e{epochs}/<scheme>/<class>/` | Generated PNGs for FID                                       |
| `experiment_fid_per_class.csv`                    | Per-class FID log                                            |
| `paper_outputs/h{size}_e{epochs}/`                | `tables/*.csv`, `figures/*.png` (loss curves, qualitative grids) |

---

## 3. Train single models manually (optional)

Each script is self-contained; defaults use `./multigames_datasets` and write under `log/`.

### Class-conditional DDPM — [DDPM](https://arxiv.org/abs/2006.11239) + class labels

```bash
# baseline | min_adagn | baseline_attn
python conditional_ddpm.py --arch baseline --img_size 64 --epochs 100 --data_root ./multigames_datasets
```

### Classifier-free guidance — [CFG](https://arxiv.org/abs/2207.12598)

```bash
python classifier_free_ddpm.py --backbone baseline --img_size 64 --epochs 100 --data_root ./multigames_datasets
```

### Classifier-guided — [Dhariwal & Nichol 2021](https://arxiv.org/abs/2105.05233)

Trains **unconditional DDPM** then **noise classifier** (`train_all` mode):

```bash
python guided_ddpm.py --mode train_all --backbone baseline --img_size 64 --epochs 100 --data_root ./multigames_datasets
```

Sample for FID (example paths; see `gamescene_schemes.ckpt_paths_guided`):

```bash
python sample_for_fid.py \
  --scheme guided_baseline \
  --img_size 64 \
  --match_data_root ./multigames_datasets \
  --out_dir gen_fid_all/h64_e100/guided_baseline \
  --ckpt_eps log/guided_unconditional_baseline_64_e100/ddpm_game_uncond_epochs100.pth \
  --ckpt_classifier log/guided_classifier_baseline_64_e100/classifier_guided_epochs100.pth \
  --grad_scale 0.2 \
  --device cuda:0
```

Scheme names and checkpoint paths are centralized in `gamescene_schemes.py`.

---

## Scheme naming (short aliases)

`gamescene_schemes.parse_experiment_scheme()` accepts full names or legacy shortcuts:

| Alias           | Full name                   |
| --------------- | --------------------------- |
| `baseline`      | `conditional_baseline`      |
| `min_adagn`     | `conditional_min_adagn`     |
| `baseline_attn` | `conditional_baseline_attn` |
| `cfg`           | `cfg_baseline`              |
| `guided`        | `guided_baseline`           |

---

## References

- DDPM: Ho et al., 2020 — https://arxiv.org/abs/2006.11239  
- Classifier guidance: Dhariwal & Nichol, 2021 — https://arxiv.org/abs/2105.05233  
- Classifier-free guidance: Ho & Salimans, 2022 — https://arxiv.org/abs/2207.12598  

---

## License / data

Procgen environments and game assets are subject to their own licenses. This repository contains only code and example result CSVs; **datasets and checkpoints are not included** (see `.gitignore`).
