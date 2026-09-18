"""SB3 загварыг GenPent-ийн агент интерфейс рүү оруулах адаптер.

evaluate_agent() нь choose_action(obs) интерфейстэй агент хүлээдэг тул
SB3-ийн on-policy/off-policy загваруудыг энэ адаптерээр ороож үнэлэнэ.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class Sb3Agent:
    """SB3 BaseAlgorithm → choose_action(obs) адаптер.

    MaskablePPO-ийн хувьд одоогийн action mask-г predict-д дамжуулна —
    үгүй бол greedy argmax нь invalid action-г сонгож loop-д ордог.
    """

    def __init__(self, model: Any, env: Any = None, deterministic: bool = True) -> None:
        """Адаптер үүсгэх.

        Args:
            model: SB3 загвар (DQN, PPO, MaskablePPO...)
            env: action mask авах орчин (MaskablePPO-д шаардлагатай)
            deterministic: үнэлгээнд greedy policy ашиглах эсэх
        """
        self.model = model
        self.env = env
        self.deterministic = deterministic
        self._use_masks = model.__class__.__name__ == "MaskablePPO"

    def choose_action(self, obs: np.ndarray) -> int:
        """Observation-д үндэслэн SB3 загвараар action сонгоно."""
        if self._use_masks and self.env is not None:
            mask = self.env.get_action_mask().astype(np.float32)
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic, action_masks=mask.reshape(1, -1)
            )
            return int(action)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)
