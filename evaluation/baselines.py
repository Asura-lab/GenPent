"""Baseline агентуудын үнэлгээний CLI — олон scenario, олон агент, CSV тайлан."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Скриптийг шууд ажиллуулахад төслийн үндсэн хавтасыг import path-д нэмэх
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from agents.bruteforce_agent import BruteForceAgent
from agents.random_agent import RandomAgent
from evaluation.evaluate import evaluate_agent

logger = logging.getLogger(__name__)


def run_baseline(
    scenario: str,
    agent_name: str,
    n_episodes: int,
    seed: int,
    fully_obs: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Нэг scenario + агент хослолд үнэлгээ ажиллуулж (summary, episode df) буцаана.

    Args:
        scenario: NASim benchmark нэр (tiny, small, medium, large, tiny-honeypot, small-honeypot)
        agent_name: "random" эсвэл "bruteforce"
        n_episodes: episode-ийн тоо
        seed: суурь seed
        fully_obs: бүрэн ажиглагдахуйц горим (debug-д хэрэгтэй)
    """
    from nasim import make_benchmark

    env = make_benchmark(scenario, fully_obs=fully_obs, flat_actions=True, flat_obs=True)

    if agent_name == "random":
        agent = RandomAgent(env, mode="masked", seed=seed)
    elif agent_name == "bruteforce":
        agent = BruteForceAgent(env)
    else:
        raise ValueError(f"Танигдахгүй агент: {agent_name}")

    df = evaluate_agent(env, agent, n_episodes=n_episodes, seed=seed)
    env.close()

    summary = {
        "scenario": scenario,
        "agent": agent_name,
        "n_episodes": n_episodes,
        "success_rate": df["success"].mean(),
        "mean_reward": df["reward"].mean(),
        "std_reward": df["reward"].std(),
        "mean_steps": df["steps"].mean(),
        "median_steps": df["steps"].median(),
        "mean_discovery": df["discovery_ratio"].mean(),
        "mean_access": df["access_ratio"].mean(),
        "mean_sensitive_owned": df["sensitive_owned_ratio"].mean(),
        "seed": seed,
        "fully_obs": fully_obs,
    }
    return summary, df


def main() -> None:
    """CLI — олон scenario/агент давтаж CSV хадгална."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="GenPent baseline үнэлгээ")
    parser.add_argument("--scenarios", nargs="+", default=["tiny", "small"])
    parser.add_argument("--agents", nargs="+", default=["random", "bruteforce"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/baselines.csv")
    args = parser.parse_args()

    all_summaries: list[dict] = []
    all_episodes: list[pd.DataFrame] = []

    for scenario in args.scenarios:
        for agent_name in args.agents:
            logger.info(f"=== {agent_name} on {scenario} ({args.episodes} episodes) ===")
            summary, df = run_baseline(
                scenario=scenario,
                agent_name=agent_name,
                n_episodes=args.episodes,
                seed=args.seed,
            )
            all_summaries.append(summary)
            df.insert(0, "agent", agent_name)
            df.insert(0, "scenario", scenario)
            all_episodes.append(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries).to_csv(out_path, index=False)
    episodes_path = out_path.with_name(out_path.stem + "_episodes.csv")
    pd.concat(all_episodes, ignore_index=True).to_csv(episodes_path, index=False)

    print(pd.DataFrame(all_summaries).to_string(index=False))
    logger.info(f"Үр дүн: {out_path} + {episodes_path}")


if __name__ == "__main__":
    main()
