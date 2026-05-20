"""训练过程 epoch 级 loss 记录，供 paper_assets 绘制 loss 曲线。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrainingLossLogger:
    def __init__(
        self,
        log_path: str | Path,
        *,
        scheme: str = "",
        paradigm: str = "",
        backbone: str = "",
        img_size: int = 0,
        phase: str = "train",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.meta = {
            "scheme": scheme,
            "paradigm": paradigm,
            "backbone": backbone,
            "img_size": img_size,
            "phase": phase,
        }
        if extra:
            self.meta.update(extra)
        if self.path.is_file():
            self._load_existing()

    def _load_existing(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.meta.update({k: v for k, v in data.items() if k != "epochs"})
                self.records = list(data.get("epochs", []))
        except (json.JSONDecodeError, OSError):
            self.records = []

    def log_epoch(self, epoch: int, loss_ema: float, **kwargs: Any) -> None:
        row: dict[str, Any] = {"epoch": int(epoch), "loss_ema": float(loss_ema)}
        row.update(kwargs)
        self.records.append(row)
        self.flush()

    def flush(self) -> None:
        payload = {**self.meta, "epochs": self.records}
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
