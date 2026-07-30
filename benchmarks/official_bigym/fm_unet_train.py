#!/usr/bin/env python3
"""Launch the official FM-UNet baseline on the shared BiGym export."""

from __future__ import annotations

import sys

from benchmarks.official_bigym.a2a_train import main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv[1:1] = ["--method", "fm_unet"]
    raise SystemExit(main())
