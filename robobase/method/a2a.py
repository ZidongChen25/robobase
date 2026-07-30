"""Plug-and-play Action-to-Action Flow Matching policy."""

from __future__ import annotations

from robobase.method.flow_matching import FlowMatching, FlowSourceSpec


class A2A(FlowMatching):
    """Flow Matching policy whose source is encoded executed-action history."""

    def __init__(self, *args, flow_source: FlowSourceSpec | None = None, **kwargs):
        flow_source = FlowSourceSpec(type="a2a") if flow_source is None else flow_source
        if flow_source.type not in {"a2a", "a2a_noise"}:
            raise ValueError("A2A requires flow_source.type=a2a or a2a_noise.")
        super().__init__(*args, flow_source=flow_source, **kwargs)


__all__ = ["A2A"]
