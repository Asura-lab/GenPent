"""Сургалтын callback — үе үе үнэлгээ, MLflow бүртгэл, best checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from stable_baselines3.common.callbacks import BaseCallback

from agents.sb3_agent import Sb3Agent
from evaluation.evaluate import evaluate_agent


class PeriodicEvalCallback(BaseCallback):
    """eval_freq алхам бүрт val орчин дээр үнэлгээ хийж MLflow-д бүртгэнэ.

    Success rate болон mean reward дээр best model-ийг checkpoint хадгална.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        checkpoint_dir: str = "models/checkpoints",
        run_name: str = "run",
        verbose: int = 0,
    ) -> None:
        """Callback үүсгэх.

        Args:
            eval_env: үнэлгээний NASim орчин
            eval_freq: хэдэн training алхам бүрт үнэлэх
            n_eval_episodes: үнэлгээ бүрт хэдэн episode ажиллуулах
            checkpoint_dir: best model хадгалах хавтас
            run_name: checkpoint файлын нэрийн угтвар
            verbose: лог хэвлэх түвшин
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        self.best_success: float = -1.0
        self.best_reward: float = -float("inf")
        self._eval_idx: int = 0

    def _on_step(self) -> bool:
        """eval_freq алхам бүрт дуудагдана — үнэлгээ + бүртгэл."""
        if self.num_timesteps % self.eval_freq != 0:
            return True

        agent = Sb3Agent(self.model, env=self.eval_env, deterministic=True)
        df = evaluate_agent(
            self.eval_env, agent, n_episodes=self.n_eval_episodes, seed=1000 + self._eval_idx
        )
        self._eval_idx += 1

        success = float(df["success"].mean())
        mean_reward = float(df["reward"].mean())
        mean_steps = float(df["steps"].mean())

        mlflow.log_metrics(
            {
                "eval_success_rate": success,
                "eval_mean_reward": mean_reward,
                "eval_mean_steps": mean_steps,
            },
            step=self.num_timesteps,
        )

        improved = success > self.best_success or (
            success == self.best_success and mean_reward > self.best_reward
        )
        if improved:
            self.best_success = max(self.best_success, success)
            self.best_reward = max(self.best_reward, mean_reward)
            path = self.checkpoint_dir / f"{self.run_name}_best"
            self.model.save(str(path))
            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps:>8,}] success={success:.2f} "
                    f"reward={mean_reward:.1f} steps={mean_steps:.0f} → best saved"
                )
        elif self.verbose > 0:
            print(
                f"[{self.num_timesteps:>8,}] success={success:.2f} "
                f"reward={mean_reward:.1f} steps={mean_steps:.0f}"
            )
        return True
