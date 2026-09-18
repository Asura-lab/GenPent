"""MaskablePPO агент GPU дээр сургах скрипт.

Action mask-ийг policy түвшинд хэрэглэдэг тул NASim-ийн холбоо барих алдаа
(reachable/discovered бус target) болон privesc-ийн хүчингүй нөхцөлд
хэзээ ч орохгүй — DQN-ийн Q-overestimation loop-оос зарчмын хувьд чөлөөт.

Ашиглалт:
    python training/train_maskppo.py --scenario tiny --timesteps 120000 --seed 42
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
from sb3_contrib import MaskablePPO

from envs.nasim_gym import NasimGymWrapper
from training.callbacks import PeriodicEvalCallback

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    """YAML тохиргооны файл уншина."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_maskppo(
    scenario: str = "tiny",
    total_timesteps: int = 120_000,
    seed: int = 42,
    device_request: str = "cuda",
    config_path: str = "configs/train_config.yaml",
    run_name: str | None = None,
    ent_coef: float | None = None,
    reward_shaping: float = 0.0,
) -> tuple[MaskablePPO, dict]:
    """MaskablePPO загварыг NASim scenario дээр сургана.

    Args:
        scenario: NASim benchmark нэр (tiny, small, medium...)
        total_timesteps: нийт сургалтын алхам
        seed: санамсаргүй төлөв
        device_request: "cuda" (анхдагч) эсвэл "cpu"
        config_path: сургалтын тохиргооны файл
        run_name: MLflow run нэр (None бол автомат)
        ent_coef: entropy coefficient override (exploration sweep-д)
        reward_shaping: прогресс bonus-ийн масштаб (0 = унтраа, жишээ: 50.0)

    Returns:
        (сургагдсан загвар, эцсийн үнэлгээний summary)
    """
    cfg = _load_yaml(config_path)
    mppo_cfg = cfg.get("maskppo", {})

    device = "cuda" if (device_request == "cuda" and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        logger.info(f"GPU ашиглаж байна: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CPU дээр ажиллаж байна!")

    # ── Орчнууд ──────────────────────────────────────────────────────────────────
    env = NasimGymWrapper(
        __import__("nasim").make_benchmark(scenario, flat_actions=True, flat_obs=True),
        reward_shaping=reward_shaping,
    )
    eval_env = NasimGymWrapper(
        __import__("nasim").make_benchmark(scenario, flat_actions=True, flat_obs=True),
        reward_shaping=reward_shaping,
    )

    # ── Загвар ───────────────────────────────────────────────────────────────────
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=mppo_cfg.get("learning_rate", 3e-4),
        n_steps=mppo_cfg.get("n_steps", 2048),
        batch_size=mppo_cfg.get("batch_size", 512),
        n_epochs=mppo_cfg.get("n_epochs", 10),
        gamma=mppo_cfg.get("gamma", 0.99),
        gae_lambda=mppo_cfg.get("gae_lambda", 0.95),
        clip_range=mppo_cfg.get("clip_range", 0.2),
        ent_coef=mppo_cfg.get("ent_coef", 0.01) if ent_coef is None else ent_coef,
        vf_coef=mppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=mppo_cfg.get("max_grad_norm", 0.5),
        verbose=0,
        seed=seed,
        device=device,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"MaskablePPO: {n_params:,} параметр | device={model.device} | seed={seed}")

    # ── Callback + MLflow ────────────────────────────────────────────────────────
    _run_name = run_name or f"MaskPPO_{scenario}_s{seed}"
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
                "algo": "MaskablePPO",
                "scenario": scenario,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "device": str(model.device),
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "n_params": n_params,
                "reward_shaping": reward_shaping,
                **{f"mppo_{k}": v for k, v in mppo_cfg.items()},
            }
        )

        logger.info(f"Сургалт эхэлж байна — {total_timesteps:,} алхам...")
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

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

    parser = argparse.ArgumentParser(description="GenPent MaskablePPO сургалт")
    parser.add_argument("--scenario", default="tiny")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--shaping", type=float, default=0.0,
                        help="Прогресс bonus-ийн масштаб (0 = унтраа, жишээ: 50)")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    for s in seeds:
        logger.info(f"=== Seed {s} ===")
        _, summary = train_maskppo(
            scenario=args.scenario,
            total_timesteps=args.timesteps,
            seed=s,
            device_request=args.device,
            config_path=args.config,
            run_name=args.run_name,
            ent_coef=args.ent_coef,
            reward_shaping=args.shaping,
        )
        logger.info(
            f"Дууслаа: {summary['run_name']} | "
            f"best success={summary['best_eval_success']:.2f} "
            f"reward={summary['best_eval_reward']:.1f}"
        )
