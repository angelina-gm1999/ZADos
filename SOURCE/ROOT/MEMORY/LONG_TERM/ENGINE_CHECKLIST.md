# LTMM Part 3 — Missing Engine Checklist

Per spec Part F: "list only, no specs yet."

## New Engines Needed

1. **IdentityConsolidationEngine** — Promotes identity-relevant MTMM packets
   to identity/core or identity/conclusions. Triggers peer-review queue for
   core memory updates.

2. **HeldThinkingBlockCaptureEngine** — Monitors emotion threshold during
   thinking phases. When threshold > 0.6, captures current thought fragment
   as a HeldThinkingBlock in thoughts/held_blocks.

3. **OverviewLogComposer** — End-of-session engine that produces an
   OverviewLogEntry summarizing mode sequence, dominant emotions, NT arc,
   and open threads.

4. **KnowledgeConsolidationEngine** — Routes validated lessons, academic
   questions, and notebook entries to the knowledge/ namespace stores.
   Interfaces with KnowledgeMapStore for graph updates.

5. **AcademicBufferMonitor** — Parallel to UnsolvedConceptsBuffer tick logic
   but for academic-domain concepts. Identifies REM Dream candidates.

6. **IdentityJournalWriter** — Produces identity journal entries from
   self-reflective pipeline output. Supports threaded replies.

7. **LibraryIngestionEngine** — Parses and stores reference material
   (books, articles, documents) into knowledge/library.

## Existing Engine Updates

1. **MemoryConsolidationEngine** — Add namespace-aware routing: identity-flagged
   packets → identity/core, learning outputs → knowledge/lessons.

2. **MemoryRelevanceHeuristicsEngine** — Extend to scan namespaced stores
   (not just flat LTMM). Identity-relevant entries never demoted across
   all stores.

3. **FractalPatternComparator** — Add cross-namespace deduplication
   (e.g., a lesson that duplicates an identity conclusion).

4. **E29 (Memory Compression)** — Override rules should check namespace:
   identity → VERBATIM, knowledge/lessons validated → SEMANTIC.

5. **E8 (Relevance Scoring)** — Namespace-aware scoring: identity entries
   get higher base relevance.

6. **E22 (Contextual Learning)** — Write context fingerprints to
   knowledge/cognitools_data instead of engine-local state.

## Pipeline Stubs Needing Specs

1. **M2 Peer Review Pipeline** — Reads PendingUpdateQueue, validates
   proposed core memory updates, writes approval/rejection.

2. **M3 Identity Check Pipeline** — Cross-references conclusions against
   core memories for consistency.

3. **M4 Knowledge Review Pipeline** — Validates lessons against
   knowledge maps and existing lessons.

4. **REM Sleep Pipeline** — Processes unsolved buffer + academic buffer
   dream candidates.

5. **Dream Pipeline** — Creative recombination of stagnated concepts
   with existing knowledge.
