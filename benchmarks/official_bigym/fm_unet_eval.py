#!/usr/bin/env python3
"""Evaluate an official FM-UNet checkpoint in the raw BiGym environment."""

from __future__ import annotations

import sys

from benchmarks.official_bigym.a2a_eval import main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv[1:1] = ["--method", "fm_unet"]
    raise SystemExit(main())
