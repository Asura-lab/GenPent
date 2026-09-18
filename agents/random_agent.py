"""Random baseline агент.

Хоёр горимтой:
  - "masked" (анхдагч): зөвхөн mask-д зөвшөөрөгдсөн action-уудаас санамсаргүй сонгоно
    — invalid action-ийн −1 торгууль авахгүй тул random-ийн жинхэнэ дээд хязгаарыг үзүүлнэ
  - "raw": бүх action-оос санамсаргүй сонгоно (NASim-ийн санал болгосон random)
"""
from __future__ import annotations

import numpy as np


class RandomAgent:
    """Action mask дээр суурилсан санамсаргүй агент."""

    def __init__(self, env, mode: str = "masked", seed: int | None = None) -> None:
        """Агент үүсгэх.

        Args:
            env: NASim эсвэл wrapper хийсэн орчин
            mode: "masked" эсвэл "raw"
            seed: санамсаргүй төлөв
        """
        self.env = env
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def choose_action(self, obs) -> int:
        """Одоогийн observation-д үндэслэн action сонгоно."""
        if self.mode == "masked" and hasattr(self.env, "get_action_mask"):
            try:
                mask = np.asarray(self.env.get_action_mask(), dtype=np.float64)
            except Exception:
                mask = None
            if mask is not None and mask.sum() > 0:
                valid = np.flatnonzero(mask > 0)
                return int(self.rng.choice(valid))
        return int(self.rng.integers(self.env.action_space.n))
