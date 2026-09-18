"""Олон NASim scenario-г эргүүлэн ашиглах ensemble орчин.

Generalization судалгааны цөм: ижил хэмжээтэй (num_hosts, num_services тогтмол)
өөр топологитой K scenario-г нэг сургалтын орчин болгох. Episode бүрт дараагийн
(эсвэл санамсаргүй) scenario руу шилжинэ — агент топологи-agnostic policy сурна.

Action/observation хэмжээ нь бүх scenario дээр ижил байх ЁСТОЙ
(ижил num_hosts/services/os/processes үед NASim-ийн хэмжээсүүд зөвшөөрөгдөнө).
"""
from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleEnv(gym.Env):
    """K NASim орчныг эргүүлэн ашигладаг Gym орчин."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        envs: list[Any],
        sampling: str = "cycle",
        seed: int | None = None,
    ) -> None:
        """Ensemble орчин үүсгэх.

        Args:
            envs: NasimGymWrapper орчнуудын жагсаалт (ижил obs/action dim-тэй)
            sampling: "cycle" (дараалан) эсвэл "random" (санамсаргүй)
            seed: санамсаргүй scenario сонголтын төлөв (random горимд)
        """
        super().__init__()
        if len(envs) == 0:
            raise ValueError("Хоосон envs жагсаалт")
        if sampling not in ("cycle", "random"):
            raise ValueError(f"sampling нь 'cycle' эсвэл 'random' байх ёстой: {sampling}")

        self.envs = list(envs)
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self._idx = 0
        self._episode = 0
        self.current_env = self.envs[0]

        # Хэмжээсүүд ижил эсэхийг шалгана
        obs_dim = self.envs[0].observation_space.shape
        act_dim = self.envs[0].action_space.shape
        for i, env in enumerate(self.envs):
            if env.observation_space.shape != obs_dim or env.action_space.shape != act_dim:
                raise ValueError(
                    f"env[{i}] хэмжээс таарахгүй: obs {env.observation_space.shape} vs {obs_dim}, "
                    f"act {env.action_space.shape} vs {act_dim} — ижил num_hosts/services ашиглана уу"
                )

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    @property
    def current_scenario_idx(self) -> int:
        """Одоогийн scenario-ийн индекс."""
        return self._idx

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Дараагийн scenario руу шилжиж, орчныг сэргээнэ."""
        if self.sampling == "cycle":
            self._idx = self._episode % len(self.envs)
        else:
            self._idx = int(self.rng.integers(len(self.envs)))
        self._episode += 1
        self.current_env = self.envs[self._idx]
        return self.current_env.reset(seed=seed, options=options)

    def step(self, action: int):
        """Одоогийн scenario дээр алхам хийнэ."""
        return self.current_env.step(int(action))

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO-ийн шаардсан нэрээр mask буцаана."""
        return self.current_env.get_action_mask()

    def get_action_mask(self) -> np.ndarray:
        """NASim агент интерфейс — ижил mask."""
        return self.current_env.get_action_mask()

    def render(self) -> None:
        """Одоогийн scenario-г render хийнэ."""
        return self.current_env.render()

    def close(self) -> None:
        """Бүх scenario-г хаана."""
        for env in self.envs:
            env.close()
