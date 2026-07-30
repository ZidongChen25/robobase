from types import SimpleNamespace

from omegaconf import OmegaConf

from robobase.envs.bigym import BiGymEnvFactory


def _step(reward):
    return SimpleNamespace(
        observation={"state": reward},
        reward=reward,
        termination=False,
        truncation=False,
        info={},
    )


def test_demo_can_terminate_at_first_success():
    cfg = OmegaConf.create({"env": {"truncate_demo_at_success": True}})
    demo = [_step(value) for value in (0.0, 0.0, 1.0, 1.0, 1.0)]

    converted = BiGymEnvFactory()._demo_to_steps(cfg, [demo])[0]

    assert len(converted) == 3
    _, reward, terminated, truncated, info = converted[-1]
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["demo"] == 1


def test_demo_legacy_tail_is_preserved_by_default():
    cfg = OmegaConf.create({"env": {}})
    demo = [_step(value) for value in (0.0, 0.0, 1.0, 1.0, 1.0)]

    converted = BiGymEnvFactory()._demo_to_steps(cfg, [demo])[0]

    assert len(converted) == 5
    _, reward, terminated, truncated, _ = converted[-1]
    assert reward == 1.0
    assert not terminated
    assert truncated
