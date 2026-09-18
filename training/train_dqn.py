"""DQN агент GPU дээр сургах үндсэн скрипт.

Ашиглалт:
    python training/train_dqn.py --scenario tiny --timesteps 100000 --seed 42
    python training/train_dqn.py --scenario small --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Скриптийг шууд ажиллуулахад төслийн үндсэн хавтасыг import path-д нэмэх
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import torch
import yaml
from stable_baselines3 import DQN

from envs.nasim_gym import NasimGymWrapper
from training.callbacks import PeriodicEvalCallback

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    """YAML тохиргооны файл уншина."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(requested: str) -> str:
    """GPU ашиглахыг шалгаж, боломжгүй бол анхааралтай CPU руу унах.

    GPU-г анхдагчаар ашиглана — 'cpu'-г тусгайлаар дуудаж байж унана.
    """
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning("CUDA боломжгүй — CPU руу унаж байна (GPU ашиглахыг хүссэн байсан!)")
        return "cpu"
    name = torch.cuda.get_device_name(0)
    logger.info(f"GPU ашиглаж байна: {name}")
    return "cuda"


def train_dqn(
    scenario: str = "tiny",
    total_timesteps: int = 100_000,
    seed: int = 42,
    device_request: str = "cuda",
    config_path: str = "configs/train_config.yaml",
    run_name: str | None = None,
    exploration_fraction: float | None = None,
    exploration_final_eps: float | None = None,
) -> tuple[DQN, dict]:
    """DQN загварыг NASim scenario дээр сургана.

    Args:
        scenario: NASim benchmark нэр (tiny, small, medium...)
        total_timesteps: нийт сургалтын алхам
        seed: санамсаргүй төлөв
        device_request: "cuda" (анхдагч) эсвэл "cpu"
        config_path: сургалтын тохиргооны файл
        run_name: MLflow run нэр (None бол автомат)
        exploration_fraction: config-ийн утгыг дарж бичих (sweep-д)
        exploration_final_eps: config-ийн утгыг дарж бичих (sweep-д)

    Returns:
        (сургагдсан загвар, эцсийн үнэлгээний summary)
    """
    cfg = _load_yaml(config_path)
    dqn_cfg = cfg.get("dqn", {})

    # CLI override-ууд (config файл олшруулахгүйн тулд)
    if exploration_fraction is not None:
        dqn_cfg["exploration_fraction"] = exploration_fraction
    if exploration_final_eps is not None:
        dqn_cfg["exploration_final_eps"] = exploration_final_eps

    device = _resolve_device(device_request)

    # ── Орчнууд ──────────────────────────────────────────────────────────────────
    env = NasimGymWrapper(
        __import__("nasim").make_benchmark(scenario, flat_actions=True, flat_obs=True)
    )
    eval_env = NasimGymWrapper(
        __import__("nasim").make_benchmark(scenario, flat_actions=True, flat_obs=True)
    )

    # ── Загвар ───────────────────────────────────────────────────────────────────
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=dqn_cfg.get("learning_rate", 1e-4),
        buffer_size=dqn_cfg.get("buffer_size", 100_000),
        batch_size=dqn_cfg.get("batch_size", 128),
        learning_starts=dqn_cfg.get("learning_starts", 1_000),
        train_freq=dqn_cfg.get("train_freq", 4),
        gradient_steps=dqn_cfg.get("gradient_steps", 1),
        target_update_interval=dqn_cfg.get("target_update_interval", 1_000),
        exploration_fraction=dqn_cfg.get("exploration_fraction", 0.3),
        exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.05),
        gamma=dqn_cfg.get("gamma", 0.99),
        verbose=0,
        seed=seed,
        device=device,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"DQN: {n_params:,} параметр | device={model.device} | seed={seed}")

    # ── MLflow ───────────────────────────────────────────────────────────────────
    _run_name = run_name or f"DQN_{scenario}_s{seed}"
    ckpt_dir = _ROOT / "models" / "checkpoints" / _run_name
    callback = PeriodicEvalCallback(
        eval_env=eval_env,
        eval_freq=cfg.get("eval_freq", 10_000),
        n_eval_episodes=cfg.get("n_eval_episodes", 10),
        checkpoint_dir=str(ckpt_dir),
        run_name=_run_name,
        verbose=1,
    )

    with mlflow.start_run(run_name=_run_name) as run:
        mlflow.log_params(
            {
                "algo": "DQN",
                "scenario": scenario,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "device": str(model.device),
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "n_params": n_params,
                **{f"dqn_{k}": v for k, v in dqn_cfg.items()},
            }
        )

        logger.info(f"Сургалт эхэлж байна — {total_timesteps:,} алхам...")
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

        # Эцсийн загвар + VecNormalize байхгүй тул зөвхөн model
        model.save(str(ckpt_dir / "final_model"))
        mlflow.log_artifact(str(ckpt_dir / "final_model.zip"), artifact_path="model")
        mlflow.log_metric("best_eval_success", callback.best_success)
        mlflow.log_metric("best_eval_reward", callback.best_reward)

    env.close()
    eval_env.close()

    summary = {
        "run_name": _run_name,
        "scenario": scenario,
        "seed": seed,
        "device": str(model.device),
        "best_eval_success": callback.best_success,
        "best_eval_reward": callback.best_reward,
    }
    return model, summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="GenPent DQN сургалт")
    parser.add_argument("--scenario", default="tiny")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Олон seed дараалуулж сургах (multi-seed туршилт)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--exploration-fraction", type=float, default=None)
    parser.add_argument("--final-eps", type=float, default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    for s in seeds:
        logger.info(f"=== Seed {s} ===")
        _, summary = train_dqn(
            scenario=args.scenario,
            total_timesteps=args.timesteps,
            seed=s,
            device_request=args.device,
            config_path=args.config,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.final_eps,
        )
        logger.info(
            f"Дууслаа: {summary['run_name']} | "
            f"best success={summary['best_eval_success']:.2f} "
            f"reward={summary['best_eval_reward']:.1f}"
        )
