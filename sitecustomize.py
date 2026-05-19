"""Project-local Python startup hooks.

This repository uses a ``src`` layout. When developers run commands directly
from the repository root without installing the package in editable mode, the
``qsp`` package would otherwise not be importable. Python imports
``sitecustomize`` automatically during startup when it is present on
``sys.path``, so this file keeps local commands working without requiring a
manual ``PYTHONPATH`` export.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
