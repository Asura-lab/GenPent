"""Generalization harness — ensemble дээр сургаж, үзээгүй топологи дээр үнэлнэ.

Судалгааны асуулт: K топологи дээр сурсан агент үзээгүй топологи руу хэр сайн
шилждэг вэ? Харьцуулалт:
  - "ensemble" агент: K scenario дээр сурсан
  - "single" агент: нэг scenario дээр сурсан (controlled baseline)
Хоёуланг нь ижил үзээгүй test scenario-ууд дээр үнэлнэ.

Ашиглалт:
    python evaluation/generalize.py --timesteps 100000 --k 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sb3_contrib import MaskablePPO

from agents.sb3_agent import Sb3Agent
from envs.ensemble_env import EnsembleEnv
from envs.nasim_gym import NasimGymWrapper
from evaluation.evaluate import evaluate_agent
from training.callbacks import PeriodicEvalCallback

logger = logging.getLogger(__name__)


def make_generated_env(net_cfg: dict, seed: int) -> NasimGymWrapper:
    """Generator-аас NASim орчин үүсгэнэ.

    Args:
        net_cfg: generalization_config-ийн network хэсэг
        seed: generator-ийн seed — топологийг тодорхойлно
    """
    import nasim

    env = nasim.generate(
        num_hosts=net_cfg["num_hosts"],
        num_services=net_cfg["num_services"],
        num_os=net_cfg.get("num_os", 2),
        num_processes=net_cfg.get("num_processes", 2),
        exploit_probs=net_cfg.get("exploit_probs", 1.0),
        step_limit=net_cfg.get("step_limit"),
        seed=seed,
        flat_actions=True,
        flat_obs=True,
    )
    return NasimGymWrapper(env)


def eval_on_unseen(
    model: MaskablePPO,
    net_cfg: dict,
    test_seed_offset: int,
    n_test_scenarios: int = 5,
    episodes_per_scenario: int = 10,
) -> pd.DataFrame:
    """Үзээгүй топологи бүр дээр тусад нь үнэлгээ хийнэ.

    Args:
        model: сургагдсан MaskablePPO
        net_cfg: network тохиргоо
        test_seed_offset: test seed = offset + i
        n_test_scenarios: хэдэн үзээгүй топологи
        episodes_per_scenario: топологи бүрт хэдэн episode

    Returns:
        Баганатууд: test_seed, success_rate, mean_reward, mean_steps
    """
    rows = []
    for i in range(n_test_scenarios):
        test_seed = test_seed_offset + i
        env = make_generated_env(net_cfg, test_seed)
        agent = Sb3Agent(model, env=env)
        df = evaluate_agent(env, agent, n_episodes=episodes_per_scenario, seed=7000 + i)
        rows.append(
            {
                "test_seed": test_seed,
                "success_rate": df["success"].mean(),
                "mean_reward": df["reward"].mean(),
                "mean_steps": df["steps"].mean(),
            }
        )
        env.close()
    return pd.DataFrame(rows)


def run_generalization(
    config_path: str = "configs/generalization_config.yaml",
    timesteps_override: int | None = None,
) -> dict:
    """Ensemble vs single харьцуулалтыг ажиллуулна.

    Args:
        config_path: generalization тохиргооны файл
        timesteps_override: config-ийн timesteps-ийг дарж бичих

    Returns:
        Summary dict (MLflow-д ч бүртгэгдэнэ)
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    net_cfg = cfg["network"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    timesteps = timesteps_override or cfg.get("timesteps", 200_000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | {net_cfg['num_hosts']} hosts, {net_cfg['num_services']} services")

    results: dict = {}

    # ── 1. Ensemble агент ────────────────────────────────────────────────────────
    k = train_cfg["k_scenarios"]
    train_seeds = list(range(train_cfg["seed_start"], train_cfg["seed_start"] + k))
    logger.info(f"Ensemble: {k} scenario (seeds {train_seeds}) дээр сургаж байна...")

    train_envs = [make_generated_env(net_cfg, s) for s in train_seeds]
    ensemble = EnsembleEnv(train_envs, sampling=train_cfg.get("sampling", "cycle"), seed=42)

    eval_env = make_generated_env(net_cfg, train_seeds[0] + 500)  # val: өөр seed
    ckpt_dir = _ROOT / "models" / "checkpoints" / "Gen_ensemble"
    callback = PeriodicEvalCallback(
        eval_env=eval_env, eval_freq=20_000, n_eval_episodes=10,
        checkpoint_dir=str(ckpt_dir), run_name="Gen_ensemble", verbose=1,
    )

    model = MaskablePPO("MlpPolicy", ensemble, verbose=0, seed=42, device=device)
    with mlflow.start_run(run_name="Gen_ensemble"):
        mlflow.log_params({
            "mode": "ensemble", "k": k, "train_seeds": str(train_seeds),
            "num_hosts": net_cfg["num_hosts"], "num_services": net_cfg["num_services"],
            "timesteps": timesteps, "device": device,
        })
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
        model.save(str(ckpt_dir / "final_model"))
    ensemble.close()
    eval_env.close()

    logger.info("Ensemble agent-ийг үзээгүй топологид үнэлж байна...")
    results["ensemble"] = eval_on_unseen(
        model, net_cfg, eval_cfg["test_seed_offset"],
        episodes_per_scenario=eval_cfg["n_eval_episodes"] // 2,
    )

    # ── 2. Single агент (controlled baseline) ────────────────────────────────────
    logger.info(f"Single: зөвхөн seed {train_seeds[0]} дээр сургаж байна...")
    single_env = make_generated_env(net_cfg, train_seeds[0])
    eval_env2 = make_generated_env(net_cfg, train_seeds[0] + 500)
    ckpt_dir2 = _ROOT / "models" / "checkpoints" / "Gen_single"
    callback2 = PeriodicEvalCallback(
        eval_env=eval_env2, eval_freq=20_000, n_eval_episodes=10,
        checkpoint_dir=str(ckpt_dir2), run_name="Gen_single", verbose=1,
    )

    model2 = MaskablePPO("MlpPolicy", single_env, verbose=0, seed=42, device=device)
    with mlflow.start_run(run_name="Gen_single"):
        mlflow.log_params({
            "mode": "single", "k": 1, "train_seeds": str([train_seeds[0]]),
            "num_hosts": net_cfg["num_hosts"], "num_services": net_cfg["num_services"],
            "timesteps": timesteps, "device": device,
        })
        model2.learn(total_timesteps=timesteps, callback=callback2, progress_bar=True)
        model2.save(str(ckpt_dir2 / "final_model"))
    single_env.close()
    eval_env2.close()

    logger.info("Single agent-ийг ижил үзээгүй топологид үнэлж байна...")
    results["single"] = eval_on_unseen(
        model2, net_cfg, eval_cfg["test_seed_offset"],
        episodes_per_scenario=eval_cfg["n_eval_episodes"] // 2,
    )

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Generalization harness")
    parser.add_argument("--config", default="configs/generalization_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--out", default="results/generalization.csv")
    args = parser.parse_args()

    results = run_generalization(args.config, args.timesteps)

    rows = []
    for mode, df in results.items():
        df.insert(0, "mode", mode)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary = out.groupby("mode")[["success_rate", "mean_reward", "mean_steps"]].mean()
    print(summary.to_string())
    logger.info(f"Дэлгэрэнгүй: {out_path}")
