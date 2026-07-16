"""Test configuration for the src-layout package."""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
SRC = str(Path(PROJECT_ROOT) / "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
