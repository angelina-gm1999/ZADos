# Pre-Development Materials

**The conceptual validation phase — before formal engineering began.**

---

## What's Here

| Document | Description |
|----------|-------------|
| `ORIGIN_REPORT.md` | The story of **valuem** — the single-file prototype that preceded ZADOS. Documents what it got right, what broke it, and the first-run event where the system detected a self-referential loop in its own output and classified it as humor. |
| `toy_version_report.md` | Case study: **parameter-sensitive behavioral divergence** in the trolley problem. Demonstrates that adjusting creativity-to-logic weight ratios produces measurably different reasoning behavior, including constraint-breaking speculative extension under high creativity settings. |
| Original valuem source code | The complete single-file prototype and its SQLite memory manager. Preserved for lineage tracing — this is not production code. |

---

## Why This Matters

Every foundational idea in ZADOS — neurochemical simulation, oscillatory modulation, structural emotion detection, reward-weighted evaluation, Socratic reasoning, symbolic latency buffers — was present in valuem in embryonic form. The prototype validated the conceptual framework before formal engineering began.

valuem's collapse was architectural, not conceptual. The system's ideas were sound, but a single-file architecture with global mutable state could not sustain the complexity the vision required. That failure was informative: it revealed the specific structural requirements (isolation, typed contracts, unidirectional data flow, canonical interfaces, testability) that ZADOS was built around.

The original structural emotion keywords from valuem's first run survive in ZADOS as the heritage fast-path in Engine 28 (Emotional Detection). The first joke is preserved in the architecture.

---

## Reading Order

1. `ORIGIN_REPORT.md` — the narrative of how the project evolved from prototype to architecture
2. `toy_version_report.md` — empirical evidence of reward-weight sensitivity and behavioral controllability
3. Source code — optional, for lineage tracing only
