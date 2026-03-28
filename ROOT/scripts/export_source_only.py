"""
Regenerate ZADOS_SOURCE_CODE.txt — source code only (no tests, no TOC).

Usage:
    cd ROOT && python scripts/export_source_only.py
"""

import os
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT.parent / "ZADOS_SOURCE_CODE.txt"


def collect_src_files():
    src_files = []
    for dirpath, dirnames, filenames in os.walk(ROOT / "src"):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".venv")]
        for f in sorted(filenames):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                src_files.append(rel)
    return src_files


def main():
    src_files = collect_src_files()
    total = len(src_files)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    sep = "=" * 80

    # Header
    lines.append(sep)
    lines.append("ZADOS - Zonal Adaptive Dynamics Operating System")
    lines.append("SOURCE CODE EXPORT (no tests)")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Source files: {total}")
    lines.append("29 Cognitive Engines + CogniTools (AtomSpace/PLN/ECAN) + Memory Layer (STMM/MTMM/LTMM) + Neurochemical Core + Reward System + Sleep/Dream Neurochemistry + Session Orchestration")
    lines.append(sep)

    # File contents — no TOC, straight into code
    for i, rel_path in enumerate(src_files, 1):
        abs_path = ROOT / rel_path
        lines.append("")
        lines.append(f"FILE {i} of {total}: {rel_path}")
        lines.append(sep)
        try:
            content = abs_path.read_text(encoding="utf-8")
            lines.append(content.rstrip())
        except Exception as e:
            lines.append(f"# ERROR reading file: {e}")
        lines.append("")
        lines.append(sep)

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported {total} source files to {OUTPUT}")


if __name__ == "__main__":
    main()
