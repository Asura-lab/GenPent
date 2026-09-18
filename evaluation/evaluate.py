"""Агентын үнэлгээ — N episode ажиллуулж статистик цуглуулна."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nasim.envs.utils import AccessLevel

logger = logging.getLogger(__name__)


def _network_summary(env: Any) -> dict:
    """Episode-ийн төгсгөлд сүлжээний төлөвөөс статистик гаргана."""
    net = env.network
    state = env.current_state

    n_addr = len(net.address_space)
    n_disc = n_acc = 0
    for addr in net.address_space:
        h = state.get_host(addr)
        n_disc += int(h.discovered)
        n_acc += int(h.access > AccessLevel.NONE)

    sensitive_addrs = list(net.sensitive_hosts.keys())
    owned = sum(
        int(state.get_host(addr).access >= AccessLevel.ROOT)
        for addr in sensitive_addrs
    )

    return {
        "n_discovered": n_disc,
        "n_accessed": n_acc,
        "discovery_ratio": n_disc / max(n_addr, 1),
        "access_ratio": n_acc / max(n_addr, 1),
        "sensitive_owned_ratio": owned / max(len(sensitive_addrs), 1),
    }


def evaluate_agent(
    env: Any,
    agent: Any,
    n_episodes: int = 20,
    seed: int = 42,
    step_limit: int | None = None,
) -> pd.DataFrame:
    """Агентыг N episode-ээр үнэлж episode бүрийн дүнг DataFrame болгоно.

    Args:
        env: NASim эсвэл NasimGymWrapper орчин
        agent: choose_action(obs) интерфейстэй агент
        n_episodes: episode-ийн тоо (episode i → seed + i)
        seed: суурь seed
        step_limit: episode бүрийн нэмэлт хязгаар (None = scenario-ийн step_limit)

    Returns:
        Баганатууд: episode, reward, steps, success, discovery_ratio,
        access_ratio, sensitive_owned_ratio, n_discovered, n_accessed
    """
    records: list[dict] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            if step_limit is not None and steps >= step_limit:
                break
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            steps += 1

        stats = _network_summary(env)
        records.append(
            {
                "episode": ep,
                "reward": total,
                "steps": steps,
                "success": bool(env.goal_reached()),
                **stats,
            }
        )

    return pd.DataFrame(records)
