"""NASim орчны Gymnasium wrapper.

NASim 0.12.0-ийн нэмэлт боломжууд:
  - get_action_mask()-ийн bug-ийг засварласан action mask
    (0.12.0-д `network.host_discovered` гэсэн байхгүй метод дуудсан байгаа тул crash)
  - invalid action-д нэмэлт penalty хэрэглэх боломж
  - info dict-д scenario-ийн нэр, дараагийн алхмын action mask
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gymnasium import Wrapper

logger = logging.getLogger(__name__)


class NasimGymWrapper(Wrapper):
    """NASimEnv → Gymnasium Wrapper: засварласан action mask + penalty + info."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: Any, invalid_action_penalty: float = 0.0) -> None:
        """Wrapper үүсгэх.

        Args:
            env: NASimEnv (flat_actions=True, flat_obs=True)
            invalid_action_penalty: invalid action сонговол reward-д нэмэх сөрөг утга
                (0 = зөвхөн NASim-ийн анхдагч −1 алхмын зардал)
        """
        super().__init__(env)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self._last_mask: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Орчинг сэргээж, info-д scenario-ийн нэр + action mask нэмнэ."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_mask = self._action_mask()
        info = dict(info)
        info["scenario_name"] = getattr(self.env.scenario, "name", "custom")
        info["action_mask"] = self._last_mask.copy()
        return obs, info

    def step(self, action: int):
        """Нэг алхам хийж, invalid action-д penalty хэрэглэнэ.

        Args:
            action: flat action index

        Returns:
            (obs, reward, terminated, truncated, info) — info-д дараагийн алхмын mask
        """
        a = int(action)
        was_invalid = self._last_mask is not None and self._last_mask[a] == 0

        obs, reward, terminated, truncated, info = self.env.step(a)
        reward = float(reward)
        if was_invalid and self.invalid_action_penalty != 0.0:
            reward += self.invalid_action_penalty

        self._last_mask = self._action_mask()
        info = dict(info)
        info["action_mask"] = self._last_mask.copy()
        return obs, reward, terminated, truncated, info

    # ── Action mask ───────────────────────────────────────────────────────────────

    def _action_mask(self) -> np.ndarray:
        """Дараагийн алхамд "өдөөж болох" action-уудын mask.

        NASim-ийн загварт дараах нөхцлүүд нь ЗААВАЛ амжилтгүй тул хасана:
          - target хүртээмжгүй (reachable=0) эсвэл нээгдээгүй (discovered=0)
            → connection_error
          - privilege escalation нь хараахан халдаагүй host дээр
            → connection_error

        Exploit нь firewall/probability шалтгаанаар амжилтгүй болох боломж
        үлдээнэ — энэ нь орчны жинхэнэ explorer-ийн бэрхшээл.
        """
        env = self.env
        n = env.action_space.n
        mask = np.zeros(n, dtype=np.int64)
        state = env.current_state

        for a_idx in range(n):
            action = env.action_space.get_action(a_idx)

            if action.is_noop():
                mask[a_idx] = 1
                continue

            host = state.get_host(action.target)
            if not (host.reachable and host.discovered):
                continue
            if action.is_privilege_escalation() and not host.compromised:
                continue

            mask[a_idx] = 1

        return mask

    def render(self) -> None:
        """NASim-ийн render-ийг шууд дамжуулна."""
        return self.env.render()
