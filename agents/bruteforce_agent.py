"""BruteForce baseline агент — зөвшөөрөгдсөн үйлдлийг ээлжлэн туршина.

NASim-ийн албан ёсны bruteforce агент нь module-level функц ашигладаг тул
энд агент объектын хэлбэрээр дахин хэрэгжүүлэв. Ээлжлэл нь masked random-аас
ялгаатай: ижил дарааллыг дараагийн episode-д үргэлжлүүлж, уламжлалт
"бүх сунгалтыг турших" дамжлагыг баталгаажуулдаг.
"""
from __future__ import annotations

import numpy as np


class BruteForceAgent:
    """Үлдэгдэл mask-д зөвшөөрөгдсөн action-уудыг ээлжлэн турших агент."""

    def __init__(self, env, seed: int | None = None) -> None:
        """Агент үүсгэх.

        Args:
            env: NASim эсвэл wrapper хийсэн орчин
            seed: fallback санамсаргүй төлөв (mask бүрэн ажиллахгүй үед)
        """
        self.env = env
        self._idx = 0
        self._rng = np.random.default_rng(seed)

    def choose_action(self, obs) -> int:
        """Mask-д зөвшөөрөгдсөн action-уудын дарааллыг ээлжлэн сонгоно."""
        try:
            mask = np.asarray(self.env.get_action_mask(), dtype=np.float64)
            valid = np.flatnonzero(mask > 0)
            if len(valid) == 0:
                return 0  # noop
            a = valid[self._idx % len(valid)]
            self._idx += 1
            return int(a)
        except Exception:
            # Mask бүрэн ажиллахгүй бол санамсаргүй (бага чухал baseline)
            return int(self._rng.integers(self.env.action_space.n))
