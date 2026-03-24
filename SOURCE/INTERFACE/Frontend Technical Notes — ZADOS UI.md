# ZADOS Frontend — Technical Notes

> Godot 4.6 visual interface for the ZADOS cognitive architecture.
> Design choices, architecture patterns, and implementation notes.

---

## 1. Why Godot, Not a Web Stack

The frontend is built in **Godot 4.6** (game engine), not React/Vue/Electron. This was a deliberate choice:

| Requirement | Web Stack | Godot |
|---|---|---|
| Force-directed graph with 500+ nodes, real-time physics, drag | Requires D3/Three.js + careful perf tuning | Native scene tree + physics, GPU-accelerated out of the box |
| WebSocket token streaming at high throughput | Browser rendering can jank on rapid DOM updates | Godot's `_process` loop handles streaming natively |
| Desktop-first (no deployment, no browser) | Electron adds ~200MB overhead + Chromium process | Single lightweight binary, no browser layer |
| Complex panel layout with animation | CSS flexbox + JS animation libraries | Godot's `Container` nodes + `Tween` system |
| Rapid prototyping of custom widgets | HTML/CSS/JS per component | GDScript + scene tree, hot-reload in editor |

**Trade-off**: No web deployment. ZADOS is a local system (Python backend + local models), so browser compatibility isn't needed.

---

## 2. Overall Architecture

### 2.1 Three-Layer Separation

```
┌─────────────────────────────────┐
│  UI Layer (Scenes + Scripts)    │  Godot nodes, rendering, user input
├─────────────────────────────────┤
│  Communication Layer            │  ZADOSClient autoload (HTTP + WS)
├─────────────────────────────────┤
│  Python Backend                 │  FastAPI wrapping ZADOS orchestrator
└─────────────────────────────────┘
```

The UI layer never constructs HTTP requests. The communication layer never manipulates UI nodes. The backend never knows about Godot.

### 2.2 Signal-Driven Data Flow

All data flows through Godot signals, not callbacks or polling:

```
User clicks Send
    → ConversationWorkspace calls ZADOSClient.stream_turn(text)
        → ZADOSClient opens WebSocket, sends {"text": ...}

WebSocket receives {"phase": 4, "token": "The..."}
    → ZADOSClient emits turn_token(4, "The...")
        → ThinkingPanel._on_turn_token(4, "The...") appends text

WebSocket receives {"complete": true, "result": {...}}
    → ZADOSClient emits turn_complete(result)
        → ConversationWorkspace finalizes AI bubble
        → RewardTab updates domain bars
        → NeurochemTab adds timeline entry
        → EnginesTab refreshes grid
```

**Why signals?**
- Workspaces are instantiated/destroyed on navigation. A workspace that doesn't exist can't receive callbacks — but it also doesn't need to, because it reconnects to signals when re-instantiated.
- Multiple panels can react to the same event without the emitter knowing about them. `turn_complete` updates 4+ panels simultaneously.
- Testing: mock `ZADOSClient` by emitting signals directly, no HTTP server needed.

### 2.3 Workspace Lifecycle

Workspaces are **loaded on demand and freed on navigation**:

```gdscript
func _switch_workspace(workspace_name: String):
    if _current_workspace:
        _current_workspace.queue_free()        # destroy old
    var scene = load("res://scenes/workspaces/" + workspace_name + ".tscn")
    _current_workspace = scene.instantiate()   # create new
    workspace_container.add_child(_current_workspace)
```

**Why not preload all 5?** Memory. The Map workspace allocates physics nodes, the Memory workspace loads LTMM data, the Dev workspace builds 5 tab panels. Keeping all in memory wastes resources for workspaces the user isn't looking at.

**Trade-off**: Switching workspaces has a brief load. Acceptable for a desktop tool.

---

## 3. The ZADOSClient Singleton

`autoloads/ZADOSClient.gd` is registered as a Godot autoload — a global singleton accessible from any script via `ZADOSClient.method()`.

### 3.1 Responsibilities

1. **Session management**: Open session, set mode, get state
2. **Turn processing**: Synchronous POST (`/process`) or streaming WebSocket (`/stream/process`)
3. **Memory CRUD**: Read/write all three tiers + LTMM sub-stores
4. **Developer tools**: Reward overrides, sleep triggers, pipeline diagnostics
5. **Signal emission**: Every response becomes a signal that workspaces connect to

### 3.2 HTTP Pattern

All HTTP calls follow the same pattern:

```gdscript
func open_session():
    var http = HTTPRequest.new()
    add_child(http)
    http.request_completed.connect(_on_session_opened.bind(http))
    http.request("http://localhost:8000/session/open", [], HTTPClient.METHOD_POST)

func _on_session_opened(result, code, headers, body, http):
    http.queue_free()
    var data = JSON.parse_string(body.get_string_from_utf8())
    emit_signal("session_opened", data)
```

Key patterns:
- `HTTPRequest` nodes are created per-call and freed on completion (no connection pooling in Godot)
- Response parsing is always `JSON.parse_string` on the body
- Every response emits a signal — the HTTP callback never updates UI directly

### 3.3 WebSocket Streaming

For turn processing, WebSocket provides real-time phase events and token streaming:

```gdscript
func stream_turn(text: String):
    _ws = WebSocketPeer.new()
    _ws.connect_to_url("ws://localhost:8000/stream/process")
    # ... on connected, send {"text": text}
    # ... on message, parse and emit phase/token/complete signals
```

The WebSocket connection is opened per-turn and closed on completion. This avoids persistent connection management while still getting streaming benefits.

**Why not SSE (Server-Sent Events)?** Godot doesn't have native SSE support. WebSocket is natively supported and bidirectional (future use: client-side interrupts mid-generation).

---

## 4. UI Component Patterns

### 4.1 MessageBubble (Factory Pattern)

`MessageBubble.gd` builds its UI tree programmatically via `initialize(role, text)`:

```gdscript
func initialize(role: String, text: String):
    _role = role
    match role:
        "USER":   _build_user_bubble(text)
        "AI":     _build_ai_bubble(text)
        "GEN":    _build_generating_bubble()
```

No `class_name` declaration. This avoids Godot's headless script parser trying to register the class globally, which causes issues with dynamically instantiated components.

**Why factory, not scene?** Message bubbles are created at runtime in arbitrary quantities. A scene file would add overhead (file I/O per bubble). Building in code is faster and more flexible — the AI bubble has a detail tray, the user bubble doesn't, and the generating bubble has a phase progress indicator.

### 4.2 Collapsible Panels (Tween Animation)

ThinkingPanel and StatsPanel animate their width:

```gdscript
const EXPANDED_W := 280.0    # ThinkingPanel (320 for StatsPanel)
const ANIM_SECONDS := 0.18

func toggle():
    var tw = create_tween()
    if _expanded:
        tw.tween_property(self, "custom_minimum_size:x", 0.0, ANIM_SECONDS)
    else:
        tw.tween_property(self, "custom_minimum_size:x", EXPANDED_W, ANIM_SECONDS)
    _expanded = !_expanded
```

The toggle buttons ("❯"/"❮") sit between the panel and the center area, always visible.

**Why tween, not instant?** Instant width changes cause jarring layout shifts in the center message area. 180ms is fast enough to feel responsive, slow enough to be visually smooth.

### 4.3 Tab Containers

StatsPanel, MemoryWorkspace, DevWorkspace, and LTMMPanel all use `TabContainer` for sub-views. Each tab's content is a separate script that:
1. Connects to relevant `ZADOSClient` signals in `_ready()`
2. Updates its own UI nodes on signal receipt
3. Disconnects signals in `_exit_tree()` (when workspace is freed)

### 4.4 Node Inspector (Map Workspace)

The GraphInspector shows properties for the selected node and allows editing:

```
┌─────────────────────────┐
│ Atom: "consciousness"   │
│ Type: ConceptNode       │
│                         │
│ TruthValue              │
│  Strength  [====  ] 0.7 │  ← slider
│  Confidence [===  ] 0.6 │  ← slider
│                         │
│ AttentionValue           │
│  STI  [42    ]          │  ← spinbox
│  LTI  [18    ]          │  ← spinbox
│                         │
│ Incoming Links (3)       │
│  ├─ InheritanceLink → X │  ← clickable
│  ├─ EvaluationLink → Y  │
│  └─ SimilarityLink → Z  │
│                         │
│ [Delete Atom] [Add Link]│
└─────────────────────────┘
```

Editing TruthValue/AttentionValue sends PATCH requests through `ZADOSClient`. Navigation (clicking a link) re-centers the graph canvas on the target node.

---

## 5. Graph Visualization (Map Workspace)

### 5.1 Force-Directed Layout

The GraphCanvas implements a force-directed physics simulation:

```
Forces:
  - Node repulsion:  F = -7000 / d²     (Coulomb-like, all pairs)
  - Edge spring:     F = K * (d - L)    (K=0.035, rest length L=110)
  - Gravity:         F = 0.007 * m      (toward center)
  - Damping:         v *= 0.80          (per frame)
```

Nodes are Godot `Node2D` children of the canvas. Physics runs in `_physics_process()` with delta-time integration. When the system stabilizes (total kinetic energy < threshold), physics pauses until a node is dragged or new nodes are added.

### 5.2 Visual Encoding

Every visual property maps to a data property:

| Visual | Data Source | Encoding |
|--------|------------|----------|
| Node shape | AtomType | Circle = Concept, Diamond = Predicate, Hexagon = Link, Square = Schema |
| Node size | STI (Short-Term Importance) | Radius ∝ sqrt(STI) |
| Node opacity | TruthValue.strength | alpha = 0.3 + 0.7 * strength |
| Node border | TruthValue.confidence | thickness = 1 + 3 * confidence |
| Node glow | LTI (Long-Term Importance) | Glow intensity ∝ LTI |
| Edge style | Relationship type | Solid = Inheritance, Dashed = Similarity, Dotted = Evaluation |
| Edge thickness | Weight | width = 1 + 2 * weight |
| Edge opacity | Weight | alpha = 0.2 + 0.8 * weight |

### 5.3 Interaction Model

| Input | Action |
|-------|--------|
| Scroll wheel | Zoom (0.08× – 6.0×) |
| Middle-drag | Pan camera |
| Left-click node | Select → show in inspector |
| Double-click node | Expand neighborhood (load connected atoms) |
| Right-click node | Context menu: Add Link, Delete, Pin/Unpin |
| Drag node | Move (temporarily pins position, physics still applies to others) |

### 5.4 Performance Considerations

- **Spatial hashing** for repulsion: Only compute repulsion between nodes within a cutoff distance, not all pairs
- **LOD**: At low zoom, nodes become simple circles (no labels, no borders), edges become thin lines
- **Lazy loading**: Only the selected map's atoms are loaded. Double-click expands one neighborhood at a time, not the entire graph.

---

## 6. Streaming UX Design

### 6.1 The Generating Bubble

When the user sends a message, a GENERATING bubble appears immediately:

```
Phase 1: ● ○ ○ ○ ○ ○    Perception...
Phase 2: ● ● ○ ○ ○ ○    Modulation...
Phase 3: ● ● ● ○ ○ ○    Dispatch...
Phase 4: ● ● ● ● ○ ○    Thinking... (tokens stream to ThinkingPanel)
Phase 5: ● ● ● ● ● ○    Evaluation...
Phase 6: ● ● ● ● ● ●    Answering... (tokens stream to this bubble)
```

On Phase 6 completion, the GENERATING bubble transforms into a regular AI bubble with detail tray.

### 6.2 Dual Streaming

Phases 4 and 6 both stream tokens, but to different targets:
- **Phase 4** (Thinking) → ThinkingPanel (left side) — italic, monospace, dimmed
- **Phase 6** (Answer) → AI MessageBubble (center) — normal text, full opacity

This lets the user see "what the system is thinking" in real time while also seeing the answer form.

### 6.3 Detail Tray

Every AI bubble has a collapsible detail tray showing pipeline metadata:

```
▼ Details
  Intent: informational_query (0.87)
  Directive: ALLOW
  Mode: Normal
  Profile: regular_input
  Emotion: curiosity (0.72)
```

This is populated from `PipelineResult` fields after `turn_complete`.

---

## 7. Memory Workspace Design

### 7.1 Three-Tier Tabs Mirror Backend Architecture

The Memory workspace uses a `TabContainer` with three tabs that directly mirror the backend's memory tiers:

```
[STMM] [MTMM] [LTMM]
```

Each tier has a distinct interaction model:

| Tier | Interaction | Refresh |
|------|-------------|---------|
| STMM | Read-only (current cycle snapshot) | Auto-refresh on `turn_complete` |
| MTMM | Read + edit context prompt | Manual refresh + auto on `turn_complete` |
| LTMM | Full CRUD across 5 sub-stores | Manual refresh per sub-tab |

### 7.2 LTMM Sub-Tab Design

The LTMM tab nests 5 sub-tabs, each connecting to different LTMM namespace stores:

```
LTMM:  [Knowledge] [Journal] [Thoughts] [Identity] [Unsolved]
```

This mirrors the backend's `IdentityNamespace`, `ThoughtsNamespace`, `KnowledgeNamespace` + `JournalStore` + `UnsolvedBuffer`.

**Knowledge** is the most complex sub-tab:
- Library panel: file upload → POST `/memory/ltmm/knowledge/library`, search bar, document list
- Notebook: rich text entries, manual + AI-generated
- Knowledge Maps: opens Map workspace for the selected map
- Lessons: read-only list of pipeline-generated lessons
- Academic Buffer: stagnation tracking, dream candidate flagging

**Unsolved** has unique interactive features:
- Cards sorted by urgency → age → stagnation count
- Purple border = dream candidate (stagnation threshold exceeded)
- Three action buttons per card:
  - **Mark Resolved**: PATCH status, remove from queue
  - **Send to Chat**: copy question text to conversation input
  - **Send to Self-Reflective**: POST to `/reflective` with the question as seed

---

## 8. Developer Workspace

### 8.1 Reward System Panel — Override Flow

The Dev workspace lets developers override domain weights at runtime:

```
Normal flow:
  Mode → profile_for_mode() → static RewardProfile → fixed domain weights

Override flow:
  Dev slider → POST /dev/reward → backend uses override weights → until "Reset"
```

This is for experimentation — see how the system behaves when innovation weight is doubled, or ethics is zeroed out. Changes don't persist across sessions.

### 8.2 Sleep/Dream Panel — Manual Triggers

In normal operation, sleep cycles trigger automatically. The Dev panel provides manual triggers:

- **REM button**: Triggers a full REM cycle with 4-phase streaming progress
- **Dream button**: Triggers dream mode on a selected unsolved concept
- **Triage button**: Light NREM — just memory triage without full consolidation

Each shows real-time metrics: consolidation depth, dream permissiveness, narrative plasticity, sigma band activity.

### 8.3 Pipeline Diagnostics

Shows the raw `PipelineResult` from the last turn as a collapsible JSON tree, plus:
- Phase timing breakdown (ms per phase — useful for finding bottlenecks)
- Engine dispatch log (which engines were selected and why)
- Error log (any exceptions or warnings from the pipeline)

---

## 9. Color System

### 9.1 Base Palette

```
Background:     #141419   (very dark blue-gray)
Surface:        #1E1E24   (panels, bubbles)
Text primary:   #D9E0EB   (light gray-blue)
Text secondary: #8890A0   (labels, timestamps)
Border:         #2A2A35   (subtle panel borders)
```

### 9.2 Semantic Colors

**Modes** (background tint on mode badges):
```
Normal:          gray      #6B7280
Learning M1-M5:  blue      #3B82F6
Sleep/REM:       indigo    #6366F1
Dream:           purple    #8B5CF6
Homework:        amber     #F59E0B
Reflective:      cyan      #06B6D4
Self-Reflective: teal      #14B8A6
```

**Directives** (in reward tab + detail tray):
```
ALLOW:     green    #22C55E
SUPPRESS:  red      #EF4444
ABSTAIN:   yellow   #EAB308
```

**Reward domains** (score bars):
```
Logic:           blue      #60A5FA
Ethics:          green     #4ADE80
Innovation:      orange    #FB923C
Attunement:      purple    #A78BFA
```

**Engine grid** (tile borders):
```
Ran this turn:   green     #22C55E (border)
Skipped:         gray      #374151 (border)
```

### 9.3 NT Heatmap Colors

The NeurochemTab uses a gradient heatmap for NT concentrations:

```
0.0 (depleted)  → dark blue   #1E3A5F
0.5 (baseline)  → neutral     #2A2A35
1.0 (saturated) → bright red  #EF4444
```

Values below baseline are cool-toned, above baseline are warm-toned. This makes it immediately visible which NTs are elevated vs suppressed.

---

## 10. Current Limitations & Planned Work

### What's Not Yet Wired

1. **Backend server**: The FastAPI bridge wrapping the ZADOS orchestrator library is still under development. The frontend is ready to connect but currently has no server to talk to.
2. **Persistence**: Workspace state (panel positions, zoom level, selected tabs) doesn't persist between app restarts.
3. **Error handling**: Network failures show in console but don't have user-facing error states (e.g., "Backend unreachable" banner).

### Known Design Debt

1. **DevWorkspace builds UI in code**: Unlike other workspaces that use `.tscn` scenes, the Dev workspace constructs its entire 5-tab layout in GDScript. This was faster to iterate on but makes layout changes harder to visualize in the Godot editor.
2. **No class_name on MessageBubble**: Avoids Godot parsing issues but means no type safety when working with bubble references.
3. **Per-call HTTPRequest nodes**: Godot doesn't have connection pooling, so every HTTP call creates and destroys an `HTTPRequest` node. This is fine for the current request rate but could become an issue if polling is added.

### Planned Additions

- Notification system for sleep cycle triggers and unsolved concept alerts
- Theme switcher (dark/light)
- Layout persistence (save/restore panel states)
- Accessibility improvements (keyboard navigation within workspaces)
