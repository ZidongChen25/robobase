def __getattr__(name):
    if name == "BC":
        from robobase.method.bc import BC
        return BC
    if name == "Diffusion":
        from robobase.method.diffusion import Diffusion
        return Diffusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BC", "Diffusion"]
