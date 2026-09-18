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

    def __init__(
        self,
        env: Any,
        invalid_action_penalty: float = 0.0,
        reward_shaping: float = 0.0,
    ) -> None:
        """Wrapper үүсгэх.

        Args:
            env: NASimEnv (flat_actions=True, flat_obs=True)
            invalid_action_penalty: invalid action сонговол reward-д нэмэх сөрөг утга
                (0 = зөвхөн NASim-ийн анхдагч −1 алхмын зардал)
            reward_shaping: прогресс shaping-ийн масштаб (0 = унтраа).
                Потенцид суурилсан: reward += scale × (progress_now − progress_prev).
                Optimal policy-г алдагдуулахгүй (potential-based shaping).
        """
        super().__init__(env)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.reward_shaping = float(reward_shaping)
        self._last_mask: np.ndarray | None = None
        self._progress_prev: float = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Орчинг сэргээж, info-д scenario-ийн нэр + action mask нэмнэ."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_mask = self._action_mask()
        self._progress_prev = self._progress_score()
        info = dict(info)
        info["scenario_name"] = getattr(self.env.scenario, "name", "custom")
        info["action_mask"] = self._last_mask.copy()
        info["progress"] = self._progress_prev
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

        # Reward shaping: прогресс өссөн хэмжээгээр bonus (potential-based)
        progress = self._progress_score()
        if self.reward_shaping != 0.0:
            reward += self.reward_shaping * (progress - self._progress_prev)
        self._progress_prev = progress

        self._last_mask = self._action_mask()
        info = dict(info)
        info["action_mask"] = self._last_mask.copy()
        info["progress"] = progress
        return obs, reward, terminated, truncated, info

    # ── Progress score (reward shaping-ийн потенциал) ────────────────────────────

    def _progress_score(self) -> float:
        """Network-ийн эзлэх хувь — 0 (эхлэл) → 1 (бүх sensitive host root).

        Potential-based shaping-ийн Φ функц:
          Φ = 0.4×(дисквер хувь) + 0.3×(хандалтын хувь) + 0.3×(root хувь)
        """
        env = self.env
        net = env.network
        state = env.current_state

        n_addr = len(net.address_space)
        n_disc = n_acc = 0
        for addr in net.address_space:
            h = state.get_host(addr)
            n_disc += int(h.discovered)
            n_acc += int(h.access > 0)  # AccessLevel.NONE = 0

        sensitive = list(net.sensitive_hosts.keys())
        owned = sum(int(state.get_host(a).access >= 2) for a in sensitive)  # ROOT = 2

        return (
            0.4 * (n_disc / max(n_addr, 1))
            + 0.3 * (n_acc / max(n_addr, 1))
            + 0.3 * (owned / max(len(sensitive), 1))
        )

    # ── Action mask ───────────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO-ийн хүлээх нэрээр mask буцаана."""
        return self._action_mask()

    def get_action_mask(self) -> np.ndarray:
        """NASim-ийн интерфейтэй нийцүүлэх public mask — агентууд шууд дуудна."""
        return self._action_mask()

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
