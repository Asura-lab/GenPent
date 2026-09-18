"""Интеграцийн smoke тест — төслийн бүх модуль хамтдаа ажиллахыг батална."""
import subprocess
import sys
from pathlib import Path

import numpy as np
from nasim import make_benchmark

from agents.random_agent import RandomAgent
from envs.nasim_gym import NasimGymWrapper

GENPENT_ROOT = Path(__file__).parent.parent


def test_full_pipeline_wrapper_and_agent():
    """Wrapper + masked random агент + статистик — бүтээгдэхүүний гол урсгал."""
    env = NasimGymWrapper(make_benchmark("tiny", flat_actions=True, flat_obs=True))
    agent = RandomAgent(env, mode="masked", seed=0)

    obs, info = env.reset(seed=0)
    assert "action_mask" in info

    total = 0.0
    done = False
    steps = 0
    while not done and steps < 300:
        valid = np.flatnonzero(info["action_mask"])
        a = int(np.random.default_rng(steps).choice(valid))
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
        steps += 1

    assert steps > 0
    env.close()


def test_baselines_cli():
    """baselines.py CLI-г жижиг тохиргоогоор шууд ажиллуулж CSV үүсэхийг шалгана."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "baselines.csv")
        result = subprocess.run(
            [
                sys.executable, str(GENPENT_ROOT / "evaluation" / "baselines.py"),
                "--scenarios", "tiny",
                "--agents", "random",
                "--episodes", "2",
                "--out", out,
            ],
            cwd=GENPENT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"CLI алдаа: {result.stderr[-1000:]}"
        assert Path(out).exists()
        episodes_csv = Path(out).with_name("baselines_episodes.csv")
        assert episodes_csv.exists()
