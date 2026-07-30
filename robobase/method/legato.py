"""Plug-and-play Legato continuation flow policy."""

from __future__ import annotations

from robobase.method.flow_matching import FlowMatching, FlowSourceSpec


class Legato(FlowMatching):
    """Flow policy with native delay-aware action-chunk continuation."""

    def __init__(self, *args, flow_source: FlowSourceSpec | None = None, **kwargs):
        flow_source = (
            FlowSourceSpec(type="legato") if flow_source is None else flow_source
        )
        if flow_source.type != "legato":
            raise ValueError("Legato requires flow_source.type=legato.")
        super().__init__(*args, flow_source=flow_source, **kwargs)


__all__ = ["Legato"]
