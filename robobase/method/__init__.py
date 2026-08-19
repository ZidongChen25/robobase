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
        from robobase.method.cqn_research import CQN

        return CQN
    if name == "CQNAS":
        from robobase.method.cqn_as_research import CQNAS

        return CQNAS
    # Pristine official-fidelity JAX port (import commit 173a01f), frozen.
    if name == "CQNOfficial":
        from robobase.method.cqn import CQN

        return CQN
    if name == "CQNASOfficial":
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
    # One-file-per-research-line CQN-AS variants (R2 refactor), each
    # subclassing the frozen pristine CQNAS.
    _cqn_as_variants = {
        "CQNASStructuredExplore": "robobase.method.cqn_as_structured_explore",
        "CQNASDenseReturn": "robobase.method.cqn_as_dense_return",
        "CQNASFrozenSupportMask": "robobase.method.cqn_as_fscqn",
        "CQNASTokenSplit": "robobase.method.cqn_as_token_split",
        "CQNASMcRct": "robobase.method.cqn_as_mc_rct",
        "CQNASProgressShaping": "robobase.method.cqn_as_progress_shaping",
        "CQNASAwr": "robobase.method.cqn_as_awr",
        "CQNASFlowPolicy": "robobase.method.cqn_as_flow_policy",
        "CQNASBcPolicy": "robobase.method.cqn_as_bc_policy",
        "CQNASTwinCritic": "robobase.method.cqn_as_twin_critic",
        "CQNASTdVariants": "robobase.method.cqn_as_td_variants",
        "CQNASGuarded": "robobase.method.cqn_as_guards",
    }
    if name in _cqn_as_variants:
        import importlib

        return getattr(importlib.import_module(_cqn_as_variants[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A2A",
    "ACT",
    "BC",
    "CQN",
    "CQNAS",
    "CQNASAwr",
    "CQNASBcPolicy",
    "CQNASDenseReturn",
    "CQNASFlowPolicy",
    "CQNASFrozenSupportMask",
    "CQNASGuarded",
    "CQNASMcRct",
    "CQNASOfficial",
    "CQNASProgressShaping",
    "CQNASStructuredExplore",
    "CQNASTdVariants",
    "CQNASTokenSplit",
    "CQNASTwinCritic",
    "CQNFlowAS",
    "CQNOfficial",
    "Diffusion",
    "DrQV2",
    "DJCQN",
    "FlowMatching",
    "Legato",
    "QChunking",
]
