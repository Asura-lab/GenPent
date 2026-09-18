"""Сургалтын модулиудын тест — Sb3Agent, callback, богино DQN сургалт."""
import numpy as np
import pytest
import torch
from nasim import make_benchmark

from agents.sb3_agent import Sb3Agent
from envs.nasim_gym import NasimGymWrapper


@pytest.fixture
def tiny_wrapped():
    """Wrapper хийсэн tiny орчин."""
    env = NasimGymWrapper(make_benchmark("tiny", flat_actions=True, flat_obs=True))
    yield env
    env.close()


def test_resolve_device_prefers_gpu():
    """GPU байгаа үед 'cuda' хүсэлт нь cuda буцаах ёстой."""
    from training.train_dqn import _resolve_device

    if not torch.cuda.is_available():
        assert _resolve_device("cuda") == "cpu"  # унах ёстой
    else:
        assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("cpu") == "cpu"


def test_sb3_agent_predicts_valid_actions(tiny_wrapped):
    """Sb3Agent-ийн сонгосон action нь action space дотор байх ёстой."""
    from stable_baselines3 import DQN

    model = DQN(
        "MlpPolicy", tiny_wrapped,
        buffer_size=100, learning_starts=10, train_freq=4,
        device="cuda" if torch.cuda.is_available() else "cpu", seed=0,
    )
    agent = Sb3Agent(model)
    obs, _ = tiny_wrapped.reset(seed=0)
    for _ in range(10):
        a = agent.choose_action(obs)
        assert 0 <= a < tiny_wrapped.action_space.n


def test_short_dqn_training_gpu(tiny_wrapped):
    """3000 алхмын богино DQN сургалт амжилттай дуусах ёстой (GPU дээр)."""
    from stable_baselines3 import DQN
    from training.callbacks import PeriodicEvalCallback

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_env = NasimGymWrapper(make_benchmark("tiny", flat_actions=True, flat_obs=True))
    model = DQN(
        "MlpPolicy", tiny_wrapped,
        buffer_size=2000, batch_size=32, learning_starts=100,
        train_freq=4, target_update_interval=200,
        device=device, seed=0,
    )
    callback = PeriodicEvalCallback(
        eval_env=eval_env, eval_freq=1000, n_eval_episodes=2,
        checkpoint_dir="models/checkpoints/test_run", run_name="test",
    )
    model.learn(total_timesteps=2000, callback=callback)
    eval_env.close()

    assert callback.best_success >= 0.0
    assert callback.best_reward > -float("inf")
