def __getattr__(name):
    if name == "ACT":
        from robobase.method.act import ACT
        return ACT
    if name == "BC":
        from robobase.method.bc import BC
        return BC
    if name == "Diffusion":
        from robobase.method.diffusion import Diffusion
        return Diffusion
    if name == "FlowMatching":
        from robobase.method.flow_matching import FlowMatching
        return FlowMatching
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ACT", "BC", "Diffusion", "FlowMatching"]
