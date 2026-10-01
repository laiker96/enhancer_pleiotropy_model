#!/usr/bin/env python3
"""Run from a checkout; never depend on another results snapshot's Python code."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / "scripts")]
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path[:3])

if __name__ == "__main__":
    from reproduction.cli import main
    main()
