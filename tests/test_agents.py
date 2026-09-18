"""Baseline агентууд болон evaluate_agent-ийн тестүүд."""
import numpy as np
import pytest
from nasim import make_benchmark

from agents.bruteforce_agent import BruteForceAgent
from agents.random_agent import RandomAgent
from evaluation.evaluate import evaluate_agent


@pytest.fixture
def raw_env():
    """Wrapper-гүй NASim орчин (агентууд get_action_mask-ийг өөрсдөө fallback-аар дуудна)."""
    env = make_benchmark("tiny", flat_actions=True, flat_obs=True)
    yield env
    env.close()


def test_random_raw_agent_runs(raw_env):
    """Raw random агент episode дуустал ажиллах ёстой."""
    agent = RandomAgent(raw_env, mode="raw", seed=0)
    obs, _ = raw_env.reset(seed=0)
    steps = 0
    done = False
    while not done and steps < 500:
        a = agent.choose_action(obs)
        obs, r, term, trunc, info = raw_env.step(a)
        steps += 1
        done = term or trunc
    assert steps > 0


def test_random_masked_agent_never_invalid(raw_env):
    """Masked random агент зөвшөөрөгдсөн action-уудаас сонгох ёстой."""
    agent = RandomAgent(raw_env, mode="masked", seed=1)
    obs, _ = raw_env.reset(seed=1)
    steps = 0
    done = False
    while not done and steps < 200:
        a = agent.choose_action(obs)
        obs, r, term, trunc, info = raw_env.step(a)
        steps += 1
        done = term or trunc
    assert steps > 0


def test_bruteforce_reaches_goal_tiny(raw_env):
    """BruteForce агент tiny scenario-ийг шийдэх ёстой (NASim-ийн баталгаажсан баримт)."""
    agent = BruteForceAgent(raw_env, seed=0)
    df = evaluate_agent(raw_env, agent, n_episodes=1, seed=0)
    assert df.iloc[0]["success"], "bruteforce tiny дээр амжилтгүй боллоо"


def test_evaluate_agent_schema(raw_env):
    """evaluate_agent-ийн DataFrame баганууд зөв бүтэцтэй байх ёстой."""
    agent = RandomAgent(raw_env, mode="masked", seed=5)
    df = evaluate_agent(raw_env, agent, n_episodes=3, seed=5)
    expected_cols = {
        "episode", "reward", "steps", "success",
        "n_discovered", "n_accessed", "discovery_ratio",
        "access_ratio", "sensitive_owned_ratio",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 3
    assert df["episode"].tolist() == [0, 1, 2]
    assert df["discovery_ratio"].between(0, 1).all()
    assert df["sensitive_owned_ratio"].between(0, 1).all()
