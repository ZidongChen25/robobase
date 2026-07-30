"""Auditable launchers for the A2A paper's RoboVerse experiments."""

from benchmarks.official_roboverse.protocol import (
    PAPER_SOURCE_COMMIT,
    PAPER_TASKS,
    get_task,
)

__all__ = ["PAPER_SOURCE_COMMIT", "PAPER_TASKS", "get_task"]
