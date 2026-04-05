# ZADOS Frontend

> **Godot 4.6** visual interface for the Zonal Adaptive Dynamics Operating System
>
> Connects to the Python backend via HTTP REST + WebSocket | Work in progress

## What It Is

A real-time dashboard and conversation interface that exposes the full internals of the ZADOS cognitive architecture — every neurotransmitter level, every engine result, every reward score, every memory tier — while also functioning as a usable chat interface.

The frontend does **not** run any cognitive logic. It renders state received from the Python backend (`FastAPI @ localhost:8000`) and sends user input back through the same channel.

---

## Quick Start

1. Open the project in **Godot 4.6+**
2. Start the Python backend (`FastAPI` server on `localhost:8000`)
3. Press F5 (or Run Project) in Godot

The `ZADOSClient` autoload connects automatically on launch.

---

## Project Structure

```
frontend/
├── project.godot                  Godot project config
├── ZADOS_FRONTEND_SPEC.txt        Full UI specification (~43KB)
├── icon.svg
│
├── autoloads/
│   └── ZADOSClient.gd            Singleton API client (HTTP + WebSocket)
│
├── scenes/
│   ├── Main.tscn                  Root scene (NavBar + workspace container + StatusStrip)
│   ├── workspaces/
│   │   ├── ConversationWorkspace.tscn
│   │   ├── MemoryWorkspace.tscn
│   │   ├── LearningWorkspace.tscn
│   │   ├── DevWorkspace.tscn
│   │   └── MapWorkspace.tscn
│   └── overlays/
│       └── SleepOverlay.tscn
│
└── scripts/
    ├── Main.gd                    Root orchestrator — nav routing, workspace swap
    ├── NavBar.gd                  Top bar — 5 workspace buttons + session label
    ├── StatusStrip.gd             Bottom bar — session status + NT pulse indicator
    │
    ├── components/
    │   └── MessageBubble.gd       Chat bubbles (user/AI/generating states)
    │
    ├── overlays/
    │   └── SleepOverlay.gd        Modal overlay for REM/Dream cycles
    │
    ├── workspaces/
    │   ├── ConversationWorkspace.gd   Chat + ThinkingPanel + StatsPanel
    │   ├── MemoryWorkspace.gd         STMM / MTMM / LTMM tabs
    │   ├── LearningWorkspace.gd       Mode selector + pipeline view
    │   ├── DevWorkspace.gd            Developer tools (5 tabs)
    │   ├── MapWorkspace.gd            Force-directed graph visualization
    │   └── WorkspacePlaceholder.gd
    │
    └── panels/
        ├── ThinkingPanel.gd           Left collapsible — Phase 4 VT stream
        ├── StatsPanel.gd              Right collapsible — 3 analytics tabs
        │
        ├── tabs/
        │   ├── RewardTab.gd           Mode, profile, domain scores, urgency
        │   ├── NeurochemTab.gd        NT heatmap + metrics timeline
        │   └── EnginesTab.gd          29-engine grid + result inspector
        │
        ├── memory/
        │   ├── STMMPanel.gd           Current-cycle compressed bundle
        │   ├── MTMMPanel.gd           Session memory + context editor
        │   ├── LTMMPanel.gd           Long-term memory (5 sub-tabs)
        │   └── ltmm/
        │       ├── IdentityPanel.gd   Core memories, development log
        │       ├── JournalPanel.gd    Cognitive journal entries
        │       ├── KnowledgePanel.gd  Library, notebook, lessons, maps
        │       ├── ThoughtsPanel.gd   Held thinking blocks, overview logs
        │       └── UnsolvedPanel.gd   Unsolved concepts queue
        │
        ├── dev/
        │   ├── RewardSystemPanel.gd      Domain overrides, mode-to-profile mapping
        │   ├── SleepDreamPanel.gd        REM/Dream activation + metrics
        │   ├── HardcodedPanel.gd         Defaults editor
        │   ├── PipelineDiagnosticsPanel.gd   Phase timing, dispatch log
        │   └── PlumbingTestPanel.gd      Direct API test interface
        │
        ├── learning/
        │   ├── ModeSelector.gd           M1-M5, Homework, Reflective, Self-Ref
        │   └── PipelineView.gd           Per-mode pipeline stages
        │
        └── map/
            ├── GraphCanvas.gd            Force-directed hypergraph renderer
            └── GraphInspector.gd         Node/link property inspector
```

---

## Workspaces

The UI is organized into **5 workspaces** (full-panel swap on nav click) plus a **sleep overlay** (modal):

### 1. Conversation (primary)

The main chat interface with two collapsible side panels:

```
┌──────────┬────────────────────────┬───────────┐
│ Thinking │     Message Feed       │   Stats   │
│ Panel    │                        │   Panel   │
│ (Phase4  │  [User bubble]         │ ┌───────┐ │
│  stream) │  [AI bubble + detail]  │ │Reward │ │
│          │  [User bubble]         │ │Neuro  │ │
│  280px   │  [Generating...]       │ │Engines│ │
│  toggle  │                        │ └───────┘ │
│  key: T  │  ┌─────────────────┐   │  320px    │
│          │  │ Input    [Send] │   │  toggle   │
│          │  └─────────────────┘   │  key: S   │
└──────────┴────────────────────────┴───────────┘
```

- **ThinkingPanel** (left): Streams Phase 4 verbalized thinking in real time. "Save" button stores the trace as a held thinking block in LTMM.
- **MessageBubble**: Three states — USER (right-aligned), AI (left-aligned + expandable detail tray), GENERATING (phase progress badges → streaming text).
- **StatsPanel** (right): Three tabs:
  - **Reward**: Mode badge, reward profile, 4 domain score bars, directive indicator (allow/suppress/abstain), urgency gauge
  - **Neurochem**: Rolling timeline of NT snapshots — per-turn heatmap (12 NTs), oscillatory bands (6), 11 metric gauges (motivation, empathy, rigidity, fatigue, etc.)
  - **Engines**: 29-tile grid (green = ran, gray = skipped), click for result inspector (entity triples, pattern lists, emotion radar, JSON tree)

### 2. Memory

Three tabs for the three memory tiers:

- **STMM**: Current compressed input bundle (raw text, intent, mode, engine weights, NT signals, safety tier)
- **MTMM**: Split pane — context prompt editor (left) + memory packet log (right) with expand-to-detail
- **LTMM**: Five sub-tabs:
  - **Knowledge**: Library (document upload + search), Notebook, Knowledge Maps, Lessons, Academic Buffer, Cognitools Data
  - **Journal**: Chronological entries (REM/Dream outputs + manual), emotion/NT snapshots, reflection questions
  - **Thoughts**: Held thinking blocks (saved VT), overview logs (session summaries), general questions
  - **Identity**: Core memories (protected), development log, alignment state
  - **Unsolved**: Priority queue sorted by urgency, stagnation tracking, dream candidate highlighting, resolve/send-to-chat actions

### 3. Learning

Split pane — mode selector (left, 220px) + pipeline view (right):

- **Mode Selector**: Radio list — Regular, M1-M5, Homework, Reflective, Self-Reflective
- **Pipeline View**: Mode-specific display:
  - M1-M5: Session objective, lesson tracker, sub-mode indicator, engine weight profile
  - Homework: Trigger button → 6-phase streaming progress (analysis → contradiction resolution → question resolution → synthesis → output)
  - Reflective: Start button → progress indicator → output preview

### 4. Dev (Developer Tools)

Five tabs:

- **Reward System**: Domain score bars + weight override sliders + directive flags + NT signals grid
- **Sleep/Dream**: REM activation (4-phase progress), Dream activation (candidate browser + creative recombination stream), Triage mode
- **Hardcoded Defaults**: Key-value editor for `core/hardcoded/defaults.py`, session-only vs disk write
- **Pipeline Diagnostics**: Last PipelineResult as JSON tree, phase timing (ms), engine dispatch log, error log
- **Plumbing Test**: Direct API endpoint testing

### 5. Map (Knowledge Graph)

Split pane — inspector (left) + force-directed graph canvas (right):

- **GraphCanvas**: Physics-based visualization of AtomSpace/KnowledgeMap hypergraphs
  - Node shape by type (circle/diamond/hexagon/square)
  - Node size ∝ STI (attention), opacity ∝ strength, border ∝ confidence, glow ∝ LTI
  - Edge style by relationship type, thickness ∝ weight
  - Scroll zoom, middle-drag pan, click select, double-click expand, right-click context menu
- **GraphInspector**: Selected atom info, TruthValue/AttentionValue editors, incoming/outgoing links
- **Toolbar**: Map selector, type filter, search, import (NL → atoms), export JSON, layout toggle

### 6. Sleep Overlay (Modal)

Covers the conversation workspace during sleep cycles:

- **REM**: 4-phase streaming progress (retroactive learning → consolidation → NT rebalancing → journal write)
- **Dream**: Candidate queue (3 priority tiers), creative recombination stream, emotional driver display, narrative plasticity gauge, scene shift button

---

## Communication Layer

All networking goes through a single autoload: `ZADOSClient.gd`

### API Endpoints (HTTP REST → `localhost:8000`)

| Category | Endpoints |
|----------|-----------|
| Session | `POST /session/open`, `POST /session/set_mode`, `GET /session/state`, `GET /session/briefing` |
| Processing | `POST /process` (synchronous turn) |
| Metrics | `GET /metrics` (neurochem snapshot) |
| Memory | `GET/POST /memory/stmm/*`, `GET/POST /memory/mtmm/*`, `GET/POST /memory/ltmm/*` |
| Developer | `POST /dev/reward`, `POST /dev/sleep/*`, `GET /dev/pipeline`, `GET/POST /dev/defaults` |
| Commanded | `POST /homework`, `POST /reflective` |

### WebSocket (Streaming)

`ws://localhost:8000/stream/process` — streaming turn processing with phase-level events:

```
→ {"text": "user message"}           Client sends

← {"phase": 1, "data": {...}}        Phase completion events
← {"phase": 4, "token": "The..."}    Phase 4 thinking tokens (stream)
← {"phase": 6, "token": "Hello..."}  Phase 6 answer tokens (stream)
← {"complete": true, "result": {...}} Full PipelineResult on finish
```

### Signal System

Workspaces don't call HTTP directly — they connect to `ZADOSClient` signals:

| Signal | Emitted When |
|--------|-------------|
| `session_opened(data)` | POST /session/open completes |
| `turn_complete(result)` | Full PipelineResult received |
| `turn_phase_updated(phase, data)` | Phase 1-6 completion event |
| `turn_token(phase, text)` | Token stream (phase 4 = thinking, 6 = answer) |
| `memory_data_received(key, data)` | Any memory GET/POST response |
| `session_mode_set(mode)` | Mode change confirmation |
| `sleep_triggered()` | REM/Dream cycle starts |
| `rem_complete()` / `dream_complete()` | Sleep cycle finishes |

---

## UI Conventions

### Color Scheme

| Element | Color |
|---------|-------|
| Background | `#141419` (very dark blue-gray) |
| Text | `#D9E0EB` (light gray-blue) |
| User bubble | Dark, right-aligned |
| AI bubble | Slightly lighter, left-aligned |

**Mode colors**:
Normal = gray, Learning = blue, Sleep = indigo, Dream = purple, Homework = amber, SelfReflective = teal

**Directive colors**:
Allow = green, Suppress = red, Abstain = yellow

**Domain colors**:
Logic = blue, Ethics = green, Innovation = orange, Attunement = purple

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `T` | Toggle thinking panel |
| `S` | Toggle stats panel |
| `Enter` | Send message (when input focused) |
| `Shift+Enter` | Newline in input |

### Panel Animation

Collapsible panels (Thinking, Stats) animate width via `Tween`:
- ThinkingPanel: 0px ↔ 280px
- StatsPanel: 0px ↔ 320px
- Animation duration: 0.18s

---

## Architecture Notes

### Why Godot (Not Web)?

- **GPU-accelerated graph rendering**: The Map workspace uses physics-based force-directed layout with real-time node dragging — this is native in Godot's scene system
- **Streaming performance**: WebSocket token streaming at high throughput without browser rendering jank
- **Desktop-first**: ZADOS runs locally (Python backend + local models). No need for browser compatibility or deployment overhead
- **Scene system**: Godot's scene/node tree maps naturally to workspace/panel/tab hierarchy

### Signal-Driven Architecture

No workspace calls HTTP directly. The flow is:

```
User action → WorkspaceScript → ZADOSClient.method() → HTTP/WebSocket
                                        ↓
ZADOSClient emits signal ← response arrives
        ↓
WorkspaceScript._on_signal(data) → updates UI nodes
```

This decouples networking from rendering. Workspaces can be tested with mock signals. Adding a new API endpoint means adding one method + one signal to `ZADOSClient`, not touching workspace code.

### Scene vs Code-Built UI

Most workspaces use `.tscn` scene files for layout, with scripts adding dynamic children. Exception: `DevWorkspace.gd` builds its entire UI in code (5 tab panels are too dynamic for static scenes).

### Factory Pattern for Bubbles

`MessageBubble.gd` uses `initialize(role, text)` → internal builder methods. No `class_name` declaration to avoid Godot's headless parsing issues with dynamically instantiated scripts.

---

## Current Status

**Working:**
- Full navigation + workspace swapping
- ZADOSClient with all HTTP endpoints + WebSocket streaming
- Conversation workspace (messages, streaming Phase 4/6, detail trays)
- Stats panel (all 3 tabs — Reward, Neurochem, Engines)
- Thinking panel with Phase 4 VT streaming + save-to-LTMM
- Memory workspace (STMM/MTMM/LTMM with 5 LTMM sub-tabs)
- Learning workspace (mode selector + pipeline view)
- Dev workspace (all 5 panels)
- Map workspace (force-directed graph + inspector)
- Sleep overlay (REM + Dream flows)
- Status strip (session info + NT pulse)

**In Progress:**
- Backend FastAPI bridge (the Python server wrapping the ZADOS orchestrator is under development)
- End-to-end integration testing with live backend
- Polish: animations, transitions, responsive layout edge cases

---

## File Reference

| File | Role |
|------|------|
| `autoloads/ZADOSClient.gd` | Singleton — all HTTP + WebSocket communication |
| `scripts/Main.gd` | Root — nav routing, workspace lifecycle |
| `scripts/NavBar.gd` | Top nav bar (5 buttons) |
| `scripts/StatusStrip.gd` | Bottom status bar |
| `scripts/components/MessageBubble.gd` | Chat bubble component |
| `scripts/workspaces/*.gd` | Workspace controllers (1 per workspace) |
| `scripts/panels/ThinkingPanel.gd` | Left collapsible panel |
| `scripts/panels/StatsPanel.gd` | Right collapsible panel |
| `scripts/panels/tabs/*.gd` | Stats tab controllers (Reward, Neurochem, Engines) |
| `scripts/panels/memory/*.gd` | Memory tier panels (STMM, MTMM, LTMM + sub-tabs) |
| `scripts/panels/dev/*.gd` | Dev tool panels (5 total) |
| `scripts/panels/learning/*.gd` | Learning mode panels |
| `scripts/panels/map/*.gd` | Graph canvas + inspector |
| `scripts/overlays/SleepOverlay.gd` | Sleep/Dream modal |
| `scenes/*.tscn` | Godot scene files (layout definitions) |
| `ZADOS_FRONTEND_SPEC.txt` | Full UI specification document |
