"""`python -m dev_ui` entry point.

Boots the ZADOS stack and drops into the developer shell.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Force UTF-8 on Windows so unicode block characters render in the console.
# Safe no-op on POSIX or when stdio is already utf-8.
for _stream in ("stdout", "stderr"):
    _s = getattr(sys, _stream, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

# Ensure src/ is on the path when running from ROOT/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from dev_ui.dev_session import build_dev_session
from dev_ui.shell import ZadosShell


def main() -> int:
    parser = argparse.ArgumentParser(prog="dev_ui", description="ZADOS developer REPL")
    parser.add_argument(
        "--no-engines", action="store_true",
        help="boot without registering the 32 cognitive engines (minimal mode).",
    )
    parser.add_argument(
        "--log-level", default="ERROR",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="root logger level (default: ERROR — keeps backend warnings quiet).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("booting ZADOS stack...")
    dev = build_dev_session(register_engines=not args.no_engines)
    print(
        f"stack ready — {len(dev.engines)}/32 engines registered"
        f" ({len(dev.stack.engine_errors)} errors).\n"
    )

    shell = ZadosShell(dev)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()  # newline
        shell.do_quit("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
