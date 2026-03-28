"""
Shared term-vector search utilities for LTMM stores.

Provides tokenization, TF computation, and cosine similarity used by
LTMMStore, JournalStore, FractalPatternComparator, and all new namespaced
stores.  Previously duplicated across four files.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    """Split *text* into lowercase alphanumeric tokens."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def term_freq(tokens: List[str]) -> Dict[str, float]:
    """Normalised term-frequency vector from a token list."""
    tf: Dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1.0
    total = max(len(tokens), 1)
    return {t: c / total for t, c in tf.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two term-frequency vectors."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
