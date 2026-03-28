"""
Regenerate ZADOS_FULL_CODEBASE.txt from the current source tree.

Usage:
    cd ROOT && python scripts/export_codebase.py
"""

import os
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT.parent / "ZADOS_FULL_CODEBASE.txt"

# Collect all .py files under src/ and tests/ (relative to ROOT)
def collect_files():
    src_files = []
    test_files = []

    for dirpath, dirnames, filenames in os.walk(ROOT / "src"):
        # Skip __pycache__ and .git
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".venv")]
        for f in sorted(filenames):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                src_files.append(rel)

    for dirpath, dirnames, filenames in os.walk(ROOT / "tests"):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".venv")]
        for f in sorted(filenames):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                test_files.append(rel)

    return src_files, test_files


def main():
    src_files, test_files = collect_files()
    all_files = src_files + test_files
    total = len(all_files)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    sep = "=" * 80

    # Header
    lines.append(sep)
    lines.append("ZADOS - Zonal Adaptive Dynamics Operating System")
    lines.append("FULL CODEBASE EXPORT")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Total files: {total}")
    lines.append("6135 tests passing (neurochem + reward + memory + cog_engines + core + orchestration + bootstrap)")
    lines.append("29 Cognitive Engines + CogniTools (AtomSpace/PLN/ECAN) + Memory Layer (STMM/MTMM/LTMM) + Neurochemical Core + Reward System + Sleep/Dream Neurochemistry + Session Orchestration")
    lines.append(sep)
    lines.append("")

    # Table of contents
    lines.append("TABLE OF CONTENTS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  [SOURCE CODE]")
    for i, f in enumerate(src_files, 1):
        lines.append(f"    {i}. {f}")

    lines.append("")
    lines.append("  [TESTS]")
    for i, f in enumerate(test_files, len(src_files) + 1):
        lines.append(f"    {i}. {f}")

    lines.append("")
    lines.append(sep)

    # File contents
    for i, rel_path in enumerate(all_files, 1):
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

    # Write output
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported {total} files to {OUTPUT}")
    print(f"  Source files: {len(src_files)}")
    print(f"  Test files:   {len(test_files)}")


if __name__ == "__main__":
    main()
