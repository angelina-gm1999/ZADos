# Technical Specifications

**Architecture documentation at three levels of depth.**

---

## What's Here

| Folder | Audience | Description |
|--------|----------|-------------|
| [`COMPREHENSIBLE_SPECS/`](COMPREHENSIBLE_SPECS/) | Anyone | Accessible overviews of each architectural layer. Written for readers who want to understand the system without reading code. **Start here.** |
| [`RESEARCH_SOURCES/`](RESEARCH_SOURCES/) | Researchers & reviewers | Theoretical lineage, academic references, and mathematical framework. Maps every ZADOS component to its intellectual foundations. |
| [`TECHNICAL_SPECS/`](TECHNICAL_SPECS/) | Engineers & auditors | Full implementation specifications. Detailed, finished, and designed for deep technical review. |

---

## How to Navigate

**For a conceptual understanding of the architecture:**
Read `COMPREHENSIBLE_SPECS/` front to back. The documents are numbered and build on each other — architecture overview first, then each layer (reward, neurochemistry, memory, cognitive engines) in detail.

**For theoretical grounding and attribution:**
Read the master reference document in `RESEARCH_SOURCES/`. It maps the full intellectual lineage — from SOAR and OpenCog to computational neuroscience, decision theory, and philosophy of mind.

**For implementation-level detail:**
The `TECHNICAL_SPECS/` folder contains the formal specifications that the code was built from. These are reference documents for reviewers who need to verify the implementation matches the design.

---

## Relationship to Source Code

The source code README (`ROOT/src/README.md`) documents the code structure, design patterns, and navigation paths through the implementation. The comprehensible specs in this folder explain the *architecture* — what the system does and why — without requiring code literacy. The technical specs bridge the two: they describe the implementation at specification level, which the source code then realizes.

Spec cross-references in the codebase (e.g. `# (spec §2.2, steps B.1-B.9)`) point to documents in the `TECHNICAL_SPECS/` folder.

















ZADos © 2025 by Angela Garcia is licensed under Creative Commons Attribution-NonCommercial 4.0 International
