def __getattr__(name):
    if name == "ACT":
        from robobase.method.act import ACT
        return ACT
    if name == "BC":
        from robobase.method.bc import BC
        return BC
    if name == "CQN":
        from robobase.method.cqn import CQN

        return CQN
    if name == "CQNAS":
        from robobase.method.cqn_as import CQNAS

        return CQNAS
    if name == "Diffusion":
        from robobase.method.diffusion import Diffusion
        return Diffusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ACT", "BC", "CQN", "CQNAS", "Diffusion"]
