def __getattr__(name):
    if name == "A2A":
        from robobase.method.a2a import A2A

        return A2A
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
    if name == "CQNFlowAS":
        from robobase.method.cqn_flow import CQNFlowAS

        return CQNFlowAS
    if name == "Diffusion":
        from robobase.method.diffusion import Diffusion

        return Diffusion
    if name == "DrQV2":
        from robobase.method.drqv2 import DrQV2

        return DrQV2
    if name == "DJCQN":
        from robobase.method.djcqn import DJCQN

        return DJCQN
    if name == "FlowMatching":
        from robobase.method.flow_matching import FlowMatching

        return FlowMatching
    if name == "Legato":
        from robobase.method.legato import Legato

        return Legato
    if name == "QChunking":
        from robobase.method.q_chunking import QChunking

        return QChunking
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A2A",
    "ACT",
    "BC",
    "CQN",
    "CQNAS",
    "CQNFlowAS",
    "Diffusion",
    "DrQV2",
    "DJCQN",
    "FlowMatching",
    "Legato",
    "QChunking",
]
