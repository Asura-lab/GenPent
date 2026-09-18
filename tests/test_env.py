"""NasimGymWrapper-ийн тестүүд."""
import numpy as np
import pytest
from nasim import make_benchmark

from envs.nasim_gym import NasimGymWrapper


@pytest.fixture
def tiny_env():
    """Tiny scenario-ийн wrapper хийсэн орчин."""
    env = make_benchmark("tiny", flat_actions=True, flat_obs=True)
    wrapped = NasimGymWrapper(env)
    yield wrapped
    wrapped.close()


def test_reset_returns_gym_api(tiny_env):
    """reset нь (obs, info) буцааж, info-д scenario нэр + mask байх ёстой."""
    obs, info = tiny_env.reset(seed=42)
    assert obs.shape == tiny_env.observation_space.shape
    assert "scenario_name" in info
    assert "action_mask" in info
    mask = info["action_mask"]
    assert mask.shape[0] == tiny_env.action_space.n
    assert mask.sum() > 0


def test_action_mask_matches_nasim_validity(tiny_env):
    """Mask-д зөвшөөрөгдсөн action нь connection_error авахгүй (reachable/discovered шалгуур)."""
    tiny_env.reset(seed=0)
    mask = tiny_env._action_mask()

    state = tiny_env.env.current_state
    for idx in np.flatnonzero(mask):
        action = tiny_env.env.action_space.get_action(int(idx))
        if action.is_noop():
            continue
        host = state.get_host(action.target)
        assert host.reachable and host.discovered, f"action {idx} maskт орсон боловч target нээгдээгүй"


def test_step_contract(tiny_env):
    """step нь 5 элементтэй tuple, reward float, mask дараагийн төлөвтэй байх ёстой."""
    obs, info = tiny_env.reset(seed=1)
    mask = info["action_mask"]
    valid = np.flatnonzero(mask)
    action = int(valid[0])

    obs2, reward, terminated, truncated, info2 = tiny_env.step(action)
    assert obs2.shape == tiny_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info2["action_mask"].shape[0] == tiny_env.action_space.n


def test_invalid_action_penalty():
    """Penalty горимд invalid action-ийн reward нь penalty-гүйгээс 2-оор бага байх ёстой."""
    env_pen = NasimGymWrapper(
        make_benchmark("tiny", flat_actions=True, flat_obs=True),
        invalid_action_penalty=-2.0,
    )
    env_base = NasimGymWrapper(
        make_benchmark("tiny", flat_actions=True, flat_obs=True),
        invalid_action_penalty=0.0,
    )
    obs, info = env_pen.reset(seed=3)
    mask = info["action_mask"]
    invalid = np.flatnonzero(mask == 0)
    if len(invalid) == 0:
        pytest.skip("invalid action байхгүй байна")
    a = int(invalid[0])

    _, r_pen, _, _, _ = env_pen.step(a)
    env_base.reset(seed=3)
    _, r_base, _, _, _ = env_base.step(a)
    assert r_pen == pytest.approx(r_base - 2.0)
    env_pen.close()
    env_base.close()


def test_reproducibility(tiny_env):
    """Ижил seed → ижил episode траектор (reward дараалал)."""
    rewards = []
    for run in range(2):
        obs, info = tiny_env.reset(seed=7)
        total = 0.0
        rng = np.random.default_rng(99)
        done = False
        while not done:
            valid = np.flatnonzero(info["action_mask"])
            a = int(rng.choice(valid))
            obs, r, term, trunc, info = tiny_env.step(a)
            total += r
            done = term or trunc
        rewards.append(total)
    assert rewards[0] == pytest.approx(rewards[1])
