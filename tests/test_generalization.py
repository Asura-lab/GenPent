"""EnsembleEnv болон generalization хэрэгслийн тестүүд."""
import pytest
import yaml
from nasim import make_benchmark

from envs.ensemble_env import EnsembleEnv
from envs.nasim_gym import NasimGymWrapper


@pytest.fixture
def tiny_envs():
    """Ижил tiny scenario-ийн 3 хуулбар (хэмжээс ижил — тестийн зориулалтаар)."""
    envs = [
        NasimGymWrapper(make_benchmark("tiny", flat_actions=True, flat_obs=True))
        for _ in range(3)
    ]
    yield envs
    for env in envs:
        env.close()


def test_ensemble_same_dims(tiny_envs):
    """Ижил хэмжээсийн орчнуудаас ensemble үүсгэх ёстой."""
    ens = EnsembleEnv(tiny_envs)
    assert ens.observation_space.shape == tiny_envs[0].observation_space.shape
    assert ens.action_space.n == tiny_envs[0].action_space.n
    ens.close()


def test_ensemble_dim_mismatch():
    """Өөр хэмжээсийн орчнууд ValueError өгөх ёстой."""
    tiny = NasimGymWrapper(make_benchmark("tiny", flat_actions=True, flat_obs=True))
    small = NasimGymWrapper(make_benchmark("small", flat_actions=True, flat_obs=True))
    with pytest.raises(ValueError, match="хэмжээс"):
        EnsembleEnv([tiny, small])
    tiny.close()
    small.close()


def test_ensemble_cycle_rotation(tiny_envs):
    """Cycle горимд episode бүр дараагийн scenario руу шилжих ёстой."""
    ens = EnsembleEnv(tiny_envs, sampling="cycle")
    idxs = []
    for _ in range(6):
        ens.reset()
        idxs.append(ens.current_scenario_idx)
    assert idxs == [0, 1, 2, 0, 1, 2]
    ens.close()


def test_ensemble_random_sampling(tiny_envs):
    """Random горимд хүчин төхөөрөмжтэй индекс өгөх ёстой."""
    ens = EnsembleEnv(tiny_envs, sampling="random", seed=0)
    idxs = [ens.reset() is not None and ens.current_scenario_idx for _ in range(10)]
    assert all(0 <= i < 3 for i in idxs)
    ens.close()


def test_ensemble_step_and_mask(tiny_envs):
    """Step нь идэвхтэй scenario дээр ажиллаж, mask зөв хэмжээтэй байх ёстой."""
    ens = EnsembleEnv(tiny_envs, sampling="cycle")
    obs, info = ens.reset()
    assert obs.shape == ens.observation_space.shape
    mask = ens.action_masks()
    assert mask.shape[0] == ens.action_space.n
    obs2, r, term, trunc, info2 = ens.step(0)
    assert obs2.shape == ens.observation_space.shape
    ens.close()


def test_ensemble_empty_raises():
    """Хоосон жагсаалт ValueError өгөх ёстой."""
    with pytest.raises(ValueError, match="Хоосон"):
        EnsembleEnv([])


def test_generated_envs_same_dims():
    """Generator-ийн өөр seed-үүд ижил хэмжээстэй байх ёстой (ижил params үед)."""
    import nasim

    with open("configs/generalization_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    net = dict(cfg["network"])
    net["num_hosts"] = 5  # тестийг хурдан байлгах
    net["step_limit"] = 50

    envs = []
    for seed in [0, 1, 2]:
        env = NasimGymWrapper(
            nasim.generate(
                num_hosts=net["num_hosts"], num_services=net["num_services"],
                num_os=net.get("num_os", 2), num_processes=net.get("num_processes", 2),
                exploit_probs=net.get("exploit_probs", 1.0),
                step_limit=net.get("step_limit"), seed=seed,
                flat_actions=True, flat_obs=True,
            )
        )
        envs.append(env)

    dims = {(e.observation_space.shape, e.action_space.n) for e in envs}
    assert len(dims) == 1, f"өөр seed-үүд өөр хэмжээстэй: {dims}"

    # Ensemble нь шууд ажиллах ёстой
    ens = EnsembleEnv(envs)
    obs, info = ens.reset()
    ens.close()
