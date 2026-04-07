# Additional Materials & Knowledge Sources

**Reference documents and source materials used during development.**

---

## What's Here

| Item | Description |
|------|-------------|
| `zadOS_concept_library_COMPLETE.txt` | The ZADOS Concept Library — 256 concepts across Layers 1–3 (Ontological Primitives, Experiential Concepts, Relational & Social Concepts). This is the ontological foundation loaded into AtomSpace (E9) at boot via the knowledge bootstrap system. Each concept is defined with aliases, dependencies, typed atom links, reward domain relevance, engine relevance, and truth-value seeds. |
| `HUMINT Operational Manual 2026` (L.G. Martin) | The analytical framework from which Engine 23's (Intention Map) behavioral classification methodology derives. Provides the structured intent classification approach adapted for computational use. |

---

## How These Materials Are Used

**Concept Library → Knowledge Bootstrap**
The concept library parser (`src/zados/bootstrap/concept_library_parser.py`) reads this file and produces structured `ConceptEntry` objects. The bootstrap system seeds these into AtomSpace as typed atoms with inheritance links, evaluation links, and truth values — giving the system a shared conceptual vocabulary before its first interaction. This is what allows cognitive engines to reference a common ontology from the first turn.

**HUMINT Manual → Engine 23**
Engine 23 (Intention Map) uses a computational adaptation of the intent classification framework documented in the HUMINT manual. The connection between the HUMINT analytical methodology and Engine 23's behavioral classification is discussed in the ethics documentation (`DOCUMENTS/ETHICS_SAFETY_ALIGNMENT/`) and the open questions document (`DOCUMENTS/STATE_NEXTSTEPS_CALLFORACTION/OPENQUESTIONS_CHALLENGES_CALL_FOR_ACTION.md`, Section 4 — Dual-Use Capabilities). 
This attribution is stated explicitly for transparency.

---

## Note

These materials are included for traceability and attribution. The concept library is an active dependency of the bootstrap system. The HUMINT manual is a reference document that contextualizes the design of Engine 23. Neither is intended as standalone reading — they are source materials that the architecture draws from.




ZADos © 2025 by Angela Garcia is licensed under Creative Commons Attribution-NonCommercial 4.0 International
