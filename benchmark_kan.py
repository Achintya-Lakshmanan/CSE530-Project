#!/usr/bin/env python3
"""Entry point for perf runs targeting the KAN model."""

from __future__ import annotations

import sys

from timing_only import main

if __name__ == "__main__":
    args = ["--model", "kan"] + sys.argv[1:]
    main(args)
