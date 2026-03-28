"""
ZADOS CogniTools — Python Cognitive Toolkit
============================================

Pure-Python replacements for OpenCog Hyperon's cognitive tools,
adapted to the ZADOS architecture with neurochemical modulation.

This package contains **development-time cognitive tooling** — the
Python substitutes for AtomSpace, PLN, ECAN, etc. — as opposed to
``py_engines/`` which holds the runtime cognitive engines the AI
uses during its processing pipeline.

Modules
-------
- ``atomspace_engine``  — Engine 9: Typed Hypergraph Knowledge Store
- ``pln_engine``        — Engine 10: Probabilistic Logic Networks
- ``ecan_engine``       — Engine 16: Economic Attention Networks
- ``journal_tool``      — Reusable journaling cognitool (no storage baked in)
"""
from __future__ import annotations
