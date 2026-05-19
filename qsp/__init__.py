"""Compatibility shim for the local ``src`` package layout.

The canonical source lives under ``src/qsp``. This top-level package exists so
that local commands run from the repository root can import ``qsp`` without
requiring ``pip install -e .`` or a manual ``PYTHONPATH`` setting.
"""

from __future__ import annotations

from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_SRC_PKG_DIR = _PKG_DIR.parent / "src" / "qsp"

if not _SRC_PKG_DIR.exists():
    raise ImportError(f"Expected source package directory not found: {_SRC_PKG_DIR}")

__path__ = [str(_SRC_PKG_DIR)]
