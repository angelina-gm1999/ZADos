"""ZADOS Developer REPL.

Terminal interface for exercising every pipeline, inspecting state, and
debugging memory tiers without the Godot frontend.

Run with:  python -m dev_ui
"""
import os as _os
import sys as _sys

# Make `import zados.*` work when running from ROOT/.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.join(_os.path.dirname(_HERE), "src")
if _os.path.isdir(_SRC) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

from dev_ui.dev_session import DevSession  # noqa: E402
from dev_ui.shell import ZadosShell  # noqa: E402

__all__ = ["DevSession", "ZadosShell"]
