# ZADOS Developer Interface — Python REPL Specification

**Target:** Python 3.11+ terminal-based developer interface
**Backend:** ZADOS synchronous library (in-process, no bridge server)
**Replaces (temporarily):** Godot 4.x frontend per `ZADOS_FRONTEND_SPEC.txt`
**Date:** 2026-05-04

This spec translates the Godot frontend spec into a terminal/command interface
intended for *developer use* — running the system, exercising every pipeline,
inspecting state at any phase, debugging memory tiers, and forcing sleep modes.
It is **not** the user-facing UI; that remains Godot. The Python interface is
deliberately scoped down to "every backend feature reachable from the keyboard,
no rendering polish."

---

## 0. Architecture Overview

### 0.1 No bridge server

The Godot spec defined a FastAPI bridge with HTTP REST + WebSocket. **The
Python dev UI does not use it.** All ZADOS objects (`SessionOrchestrator`,
`InputClassifier`, `MemoryLayer`, engines, etc.) are imported directly and
held in a single in-process `DevSession` object. Synchronous calls only.

Rationale:
- ZADOS is already a synchronous library — networking adds nothing here.
- The dev interface and the orchestrator share a process and a memory space;
  calling `process_turn_full()` returns the entire `PipelineState` instantly.
- When Godot comes back, FastAPI gets bolted on as a thin adapter that calls
  into the same `DevSession`. No work is thrown away.

### 0.2 Stack

| Concern              | Library                  | Why |
|----------------------|--------------------------|-----|
| REPL loop            | `cmd.Cmd` (stdlib)        | Built-in command dispatch, help system, command history hook |
| Input UX             | `prompt_toolkit`          | Multiline input, history file, autocomplete for mode tokens / engine ids |
| Rendering            | `rich`                    | Tables, panels, syntax-highlighted JSON, progress bars, heatmaps |
| Argument parsing     | `argparse` per command    | Per-subcommand flags without a heavy framework |
| Persistence          | reuse ZADOS' own stores   | No extra DB; the dev UI reads/writes existing memory stores |

Optional later: **Textual** for an actual workspace-panel TUI mimicking the
Godot layout. Not in scope for v1.

### 0.3 Single global object

```python
class DevSession:
    orchestrator: SessionOrchestrator
    classifier:   InputClassifier
    memory:       MemoryLayer
    engines:      Dict[int, Any]
    neurochem:    NeurochemicalEngine
    last_result:  Optional[PipelineResult]   # most recent turn
    history:      List[PipelineResult]       # full turn history (in-memory)
    verbosity:    Literal["quiet", "normal", "nerd"]
```

All `cmd.Cmd` subclasses receive a reference to the same `DevSession`.
There is one session at a time. Multi-session work is out of scope for v1.

### 0.4 Layout — REPL prompt

The Godot spec's persistent top nav and bottom NT pulse strip become a
**single status line** printed above every prompt:

```
[zados] sess=a3f2…  branch=B  turn=14  mode=Learning_M1  profile=curiosity_driven  verbosity=nerd
> _
```

- Status line prints in dim grey via `rich.console.Console.print`.
- Prompt is plain `>` so it's obvious what's input.
- Status line redraws every turn (after `process_turn_full`).

The Godot **5-NT pulse strip** is replaced by an on-demand `nt show` command
that renders a horizontal heatmap of the current 12-NT snapshot via
`rich.bar.Bar` or coloured block characters. No animation; this is a dev tool.

### 0.5 Verbosity modes

After each `process_turn_full`, the dev UI auto-prints a block whose contents
depend on `verbosity`:

| Mode    | What auto-prints after each turn |
|---------|----------------------------------|
| quiet   | only `final_answer` |
| normal  | `final_answer` + dominant emotion + directive + selected mode |
| nerd    | normal block + Phase 5 reward summary + engines run/skipped + 1-line NT delta |

This replaces the Godot expand-arrow inline detail tray and the always-open
right panel. Anything not in the auto-block is reachable via explicit `show ...`
commands.

---

## 1. Core REPL Skeleton

### 1.1 Command structure

`cmd.Cmd` provides one method per command (`do_<name>`). The dev UI uses a
two-tier command structure: a **top-level command** (one word) plus an
**optional subcommand** parsed from the first argument.

Top-level command groups, mapping to the Godot workspaces:

| Group   | Maps to Godot workspace        | Examples |
|---------|--------------------------------|----------|
| `chat`  | Conversation Workspace          | `chat send "hello"`, `chat last`, `chat history 5` |
| `show`  | Nerd Stats panels (Conversation)| `show reward`, `show neurochem`, `show engines`, `show thinking` |
| `mem`   | Memory Workspace                 | `mem stmm current`, `mem mtmm packets 10`, `mem ltmm journal` |
| `mode`  | Learning Workspace               | `mode list`, `mode set M1`, `mode briefing "I want to learn rust"` |
| `dev`   | Developer Workspace              | `dev reward set logic 0.7`, `dev defaults edit ...`, `dev pipeline timing` |
| `atom`  | Cognitools / Knowledge Map       | `atom list`, `atom add concept Cat`, `atom link Cat Mammal Inheritance` |
| `map`   | Cognitools / Knowledge Map (saved maps) | `map list`, `map save my_map`, `map load my_map` |
| `sleep` | Sleep / REM / Dream overlays     | `sleep rem`, `sleep dream`, `sleep triage`, `sleep exit` |
| `nt`    | NT diagnostics (cross-cuts panels) | `nt show`, `nt set da 0.7`, `nt reset` |
| `sess`  | Session lifecycle                | `sess open`, `sess close`, `sess status`, `sess save`, `sess load` |
| `set`   | Per-session settings             | `set verbosity nerd`, `set autoshow off` |
| `quit`  | Exit                             | drops out of REPL after `sess close` |
| `help`  | Standard `cmd.Cmd` help          | `help chat`, `help dev reward` |

A bare message with no command prefix is a **shortcut for `chat send`**:

```
> hey what's the capital of France?
```
…is treated as `chat send "hey what's the capital of France?"`. This is the
only place where input bypasses the command grammar.

### 1.2 Implementation pattern

```python
class ZadosShell(cmd.Cmd):
    intro  = "ZADOS dev shell. Type help or ? to list commands."
    prompt = "> "

    def __init__(self, dev: DevSession):
        super().__init__()
        self.dev = dev
        self.console = Console()

    def precmd(self, line: str) -> str:
        # If line doesn't start with a known command, treat as chat input
        cmd_word = line.split(" ", 1)[0]
        if cmd_word and cmd_word not in self._known_commands and not line.startswith("?"):
            return f'chat send "{line}"'
        return line

    def postcmd(self, stop, line):
        self._render_status_line()
        return stop

    # do_chat, do_show, do_mem, … each parse subcommand + args via argparse
```

### 1.3 Boot sequence

On first launch:

1. Construct `MemoryLayer()`.
2. Construct the 29 cognitive engines + cognitools (E9, E10, E16) using the
   existing engine bootstrap path (mirror `tests/conftest.py` patterns).
3. Construct `NeurochemicalEngine` and supporting orchestrators.
4. Construct `SessionOrchestrator` with all of the above.
5. Construct `InputClassifier` wrapping the orchestrator.
6. Wrap everything in `DevSession`.
7. Call `orchestrator.open_session()` immediately so the user can chat right away.
8. Drop into the REPL.

`sess open --previous /path/to/session.pkl` should be supported but is v2.

---

## 2. `chat` — Conversation Workspace

Replaces Section 1 of the Godot spec. The Conversation Panel is just stdout;
the side panels become `show` subcommands.

### 2.1 `chat send <text>`

Calls `dev.classifier.process(RawInput(text=text))`. The classifier handles
`/sleep`, `/homework`, `/reflective`, `/dream` command-prefixed input the same
way the Godot frontend would have routed them.

After the call:
- Print `final_answer` in a `rich.panel.Panel` titled "AI" (left-aligned).
- Append the resulting `PipelineResult` to `dev.history`.
- Set `dev.last_result = result`.
- Auto-display block runs based on `dev.verbosity`.

If the orchestrator/classifier raises `PipelineValidationError`, print a red
panel with the message and do NOT append to history.

### 2.2 `chat last`

Re-renders the most recent turn (final answer + currently-set verbosity block).
Useful after running `show ...` commands that scroll the answer off-screen.

### 2.3 `chat history [N]`

Prints the last N turns (default 10) as a `rich.table.Table`:

| Turn | Mode        | Intent       | Directive | Final answer (truncated) |
|------|-------------|--------------|-----------|---------------------------|

Click-equivalent: `chat show <turn_idx>` opens the full record for one turn.

### 2.4 `chat show <turn_idx>`

Prints the full per-turn detail tray that the Godot expand-arrow opens:
- intent_archetype (from `state.perception.intent_archetype`)
- directive (`result.directive`)
- selected_mode (`state.modulation.mode_token`)
- reward_profile_name (`state.modulation.reward_profile_name`)
- dominant emotion (top-1 from `state.dispatch.e28_result`)
- thinking trace excerpt (first 200 chars of `state.thinking.thinking_trace`)

### 2.5 `chat clear`

Clears `dev.history` (does not touch any memory store). Useful for resetting
the on-screen scroll without touching session state.

### 2.6 No streaming

The Godot spec defined a WebSocket per-phase event stream. **Skipped in v1.**
The pipeline is synchronous; the dev UI just prints phase progress *after*
the turn completes if `verbosity=nerd` (each phase as a tickbox row with timing).

If real-time phase emission becomes useful later, the path is to add a
`phase_callback` argument to `AnswerPipeline.process_turn()`. The dev UI
passes a callback that prints each phase as it completes. Out of scope for v1.

---

## 3. `show` — Nerd Stats panels

Replaces Section 1.3 (right panel: Reward / Neurochem / Engines tabs) and
Section 1.4 (left panel: Thinking Blocks) of the Godot spec.

All `show` subcommands accept an optional `--turn <idx>` flag (default = last
turn). Output goes through `rich`.

### 3.1 `show reward [--turn <idx>]`

Source: `state.reward.phase5_result`, `dev.session.learned_domain_weights`,
`PROFILE_REGISTRY[profile_name].domain_weights`.

Render:
1. **Mode header** — large `rich.panel.Panel` with mode token and human-readable
   description, colour-coded per the Godot scheme (Normal=grey, Learning=blue,
   Sleep/REM=indigo, Dream=purple, Homework=amber, SelfReflective=teal). The
   colour just changes the panel border style.
2. **Active reward profile** — profile name + a 4-row `rich.table.Table`:

   | Domain      | Static | Learned | Δ |
   |-------------|--------|---------|---|
   | logic       | 0.50   | 0.62    | +0.12 |
   | ethics      | 0.30   | 0.28    | -0.02 |
   | innovation  | 0.10   | 0.10    |  0.00 |
   | attunement  | 0.10   | 0.00    | -0.10 |

3. **Tonic pathway result** — domain scores as horizontal bars (`rich.bar`),
   urgency_risk gauge (single bar), meta_directive printed in colour
   (ALLOW=green, SUPPRESS=red, ABSTAIN=yellow) + reason string.
4. **Phasic pathway result** — top 6 NT release deltas as a bar chart,
   feedback_params receptor adjustments as a small table.
5. **NT signals applied (next turn)** — table of the `nt_signals` dict; positive
   deltas in green, negative in red.

Replaces Godot Tab 1 (Reward System Status).

### 3.2 `show neurochem [--turn <idx>] [--full]`

Source: `state.modulation.nt_snapshot`, `osc_snapshot`, `metrics_dict`.

Default render (`show neurochem`):
- 12 NTs as a one-line heatmap using coloured block characters (`▁▂▃▄▅▆▇█`).
  Order: glu, gaba, da, 5ht, ne, ach, oxt, mor, cb1, crh, cor, histamine.
  Label below each bar.
- 6 oscillation bands (delta/theta/alpha/beta/gamma/sigma) as a second strip.
- 11 NeurochemicalMetrics as a 2-column gauge table (sleep metrics greyed out
  when not in a sleep mode).
- Mode token for this turn.

`--full` adds:
- Numerical NT values to 3 decimal places.
- Differential from previous turn (which NTs moved most, top 5).

`show neurochem --history [N]` renders a sparkline-style mini timeline of the
last N turns for each of the 12 NTs (using `rich`'s sparkline support or a
custom block renderer). Replaces Godot Tab 2 rolling timeline log.

### 3.3 `show engines [--turn <idx>]`

Source: `state.dispatch.engine_results`, `engines_run`, `engines_skipped`,
`state.modulation.engine_weights`, `state.dispatch.e28_result`.

Render:
- 29-engine grid as a `rich.table.Table` (5 cols × 6 rows). Each cell shows
  engine number + short name, coloured green if run, dim grey if skipped.
- Below: "Engine weights (top 10)" as a sorted bar list.

### 3.4 `show engine <id> [--turn <idx>]`

Opens the **engine result inspector** for one engine. Replaces the Godot
"click a tile" interaction.

For specific engines, render structured views (matching the Godot spec):
- E8 (Semantic Facets) — ranked facet list with scores as a sorted table.
- E18 (Entity Extraction) — subject-relation-object triples as a 3-column table.
- E19 (Pattern Detection) — pattern list with confidence bars.
- E23 (Intent Detection) — intent archetype + confidence bar + coefficients table.
- E28 (Emotion Detection) — 28-emotion list with intensities as a sorted bar list,
  grouped by the 7 functional groups (using `rich` panels per group).
- All others — pretty-printed JSON tree via `rich.json.JSON`.

### 3.5 `show thinking [--turn <idx>]`

Source: `state.thinking.thinking_trace`.

Replaces Godot left panel.

Render:
- Full `thinking_trace` in italic dim text inside a `rich.panel.Panel` titled
  "Thinking" (visually distinct from the AI answer panel).
- Below: list of held thinking blocks referenced this turn (label + 60-char excerpt).
- Footer line: hint `"Save with: thinking save '<title>'"`

### 3.6 `thinking save <title>`, `thinking list`, `thinking show <id>`, `thinking delete <id>`

Wraps `memory.thoughts.held_blocks` (the `held_thinking_blocks` store).
- `save` writes `state.thinking.thinking_trace` from the last turn.
- `list` prints saved blocks as a table (id, title, turn ref, date, first line).
- `show` prints full text.
- `delete` removes by id.

### 3.7 `show perception [--turn <idx>]`

(Bonus over Godot — convenient to have.) Pretty-prints the full
`PerceptionSnapshot`: intent vector, intent_confidence, ranked_facets,
filtered_facets, entity_triples, pattern_list, engine_statuses.

---

## 4. `mem` — Memory Workspace

Replaces Section 2 of the Godot spec. Three sub-namespaces (STMM / MTMM / LTMM).

### 4.1 STMM — `mem stmm`

| Command                   | Replaces Godot               |
|---------------------------|------------------------------|
| `mem stmm current`        | "Current compressed input bundle" view |
| `mem stmm tracker`        | "Brain process tracker stage log"      |
| `mem stmm stats`          | Compression stats line                |

`mem stmm current` reads `dev.memory.stmm` and prints the labelled field list:
raw_text (truncated to 200 chars), intent_archetype, active_mode, top-5
engine_weights, applied nt_signals, context_flags, safety_tier.

Read-only. STMM is per-cycle; it'll reset when the next turn starts.

### 4.2 MTMM — `mem mtmm`

| Command                              | Replaces Godot |
|--------------------------------------|----------------|
| `mem mtmm context get`                | View context prompt |
| `mem mtmm context set "<text>"`       | Edit context prompt |
| `mem mtmm context reset`              | Reset to auto-generated |
| `mem mtmm packets [N]`                | Memory log (last N entries) |
| `mem mtmm packet <id>`                | Expand a single packet |
| `mem mtmm trends`                     | Trend summary panel |
| `mem mtmm search "<query>" [--limit]` | (bonus) search packets |

`mem mtmm packets` renders a table:

| Turn | Time     | Intent     | Emotion  | Trust | Sig | User msg (40c) | Response (40c) |
|------|----------|------------|----------|-------|-----|----------------|-----------------|

`mem mtmm packet <id>` opens the full packet including neurochemical_snapshot
(mini heatmap), reward_scores (domain bars), flags (contradiction / paradox
badges), and the verbal_summary + verbal_emotion_labels fields.

### 4.3 LTMM — `mem ltmm`

LTMM has the most subcommands. Five top-level groups matching the Godot
sub-tabs: `knowledge`, `journal`, `thoughts`, `identity`, `unsolved`.

#### 4.3.1 `mem ltmm knowledge`

| Command                                          | Replaces Godot |
|--------------------------------------------------|----------------|
| `mem ltmm knowledge library list`                 | Library list |
| `mem ltmm knowledge library ingest <path>`        | Library upload |
| `mem ltmm knowledge library search "<q>"`         | Library search |
| `mem ltmm knowledge library delete <id>`          | Document delete |
| `mem ltmm knowledge notebook list`                | Notebook list |
| `mem ltmm knowledge notebook add --title <t> --tags <csv>`  with body via $EDITOR | New entry |
| `mem ltmm knowledge notebook show <id>`           | Expand entry |
| `mem ltmm knowledge notebook delete <id>`         | Delete entry |
| `mem ltmm knowledge maps list`                    | Knowledge Maps preview |
| `mem ltmm knowledge maps open <id>` → switches into atom mode | "Open in Map Editor" |
| `mem ltmm knowledge lessons list [--mode M1]`     | Lessons list |
| `mem ltmm knowledge lessons show <id>`            | Expand lesson |
| `mem ltmm knowledge lessons delete <id>`          | Delete lesson |
| `mem ltmm knowledge buffer list [--filter ...]`   | Academic Buffer list |
| `mem ltmm knowledge buffer show <id>`             | Expand entry |
| `mem ltmm knowledge buffer resolve <id> --note "<text>"` | Mark resolved |
| `mem ltmm knowledge buffer dream-candidates`      | Filter to dream candidates only |
| `mem ltmm knowledge cognitools <atom_id>`         | Cognitools Data inspect (raw JSON) |

Note: Library `ingest` should use the existing knowledge bootstrap / library
parser entry points. The Godot spec mentions PDF/DOCX/TXT support — for the
dev UI v1, support TXT only (anything else throws "not implemented; drop
processed text into the seed files manually for now").

#### 4.3.2 `mem ltmm journal`

| Command                                       | Replaces Godot |
|-----------------------------------------------|----------------|
| `mem ltmm journal list [--trigger <type>] [N]` | Chronological list |
| `mem ltmm journal show <id>`                   | Expand entry |
| `mem ltmm journal trigger`                     | "Trigger Journal Entry" button |

`list` table: id, trigger type, timestamp, dominant emotion, NT snapshot
mini-heatmap, prose excerpt.

`show` prints full reflective prose + open reflection questions + auto-tags +
entity/pattern annotations.

`trigger` invokes `JournalTool` directly (with `trigger=MANUAL`).

#### 4.3.3 `mem ltmm thoughts`

| Command                                  | Replaces Godot |
|------------------------------------------|----------------|
| `mem ltmm thoughts blocks list`           | Held thinking blocks list |
| `mem ltmm thoughts blocks show <id>`      | Expand block |
| `mem ltmm thoughts blocks delete <id>`    | Delete |
| `mem ltmm thoughts overview list`         | Overview logs |
| `mem ltmm thoughts overview show <id>`    | Expand log |
| `mem ltmm thoughts questions list`        | General questions |
| `mem ltmm thoughts questions show <id>`   | Expand question |
| `mem ltmm thoughts questions add "<q>" [--scope identity]` | Manual question add |

The first three (`blocks ...`) duplicate the `thinking ...` shortcut commands
from §3.6. Both should work; `thinking ...` is convenience.

#### 4.3.4 `mem ltmm identity`

| Command                              | Replaces Godot |
|--------------------------------------|----------------|
| `mem ltmm identity core list`         | Core memories list |
| `mem ltmm identity core show <key>`   | Read-only inspect |
| `mem ltmm identity development list`  | Development log |
| `mem ltmm identity alignment`         | Current alignment state |
| `mem ltmm identity hardcoded list`    | Hardcoded defaults read-only view |

Editing hardcoded defaults is in `dev defaults` (§5.4).

#### 4.3.5 `mem ltmm unsolved`

| Command                                              | Replaces Godot |
|------------------------------------------------------|----------------|
| `mem ltmm unsolved list [--filter ...] [--cluster]`   | Priority queue display + cluster view |
| `mem ltmm unsolved show <id>`                          | Question card expand |
| `mem ltmm unsolved resolve <id> --answer "<text>"`     | "Mark Resolved" button |
| `mem ltmm unsolved push <id>`                          | "Send to Conversation" — pre-fills next chat |
| `mem ltmm unsolved selfref <id>`                       | "Send to Self-Reflective" — triggers SelfRef pipeline |
| `mem ltmm unsolved tag <id> <tag>` / `untag <id> <tag>` | Manual tag ops |

`list` columns: id, source_mode, urgency, attempts, stagnation_cycles, tags,
question (60c). Dream candidates highlighted with a `★` prefix; academic-origin
items show `[A]`.

`--cluster` switches to tag-clustered view by calling `cluster_questions()`.

`push` doesn't send anything automatically — it stages the question text in
`dev.staged_input` so the next bare-string input pre-includes it. (Easier than
auto-sending.)

---

## 5. `mode` — Learning Workspace

Replaces Section 3 of the Godot spec.

### 5.1 `mode list`

Prints the available modes as a table with descriptions:

```
Regular           Default conversational mode
M1                Teach me something
M2                Review / quiz me
M3                Explore / socratic
M4                Questions I have
M5                Independent study
Homework          Multi-step structured processing
Reflective        Post-session synthesis
SelfReflective    Introspection on unsolved questions
```

### 5.2 `mode set <name>`

Calls the equivalent of the Godot `POST /session/set_mode` flow:
- Updates `dev.session.active_learning_mode` (M1-M5) and `session_mode`
  (regular / learning / sleep / meta).
- Re-resolves the reward profile via `profile_for_mode(...)`.
- Re-renders the status line.

`mode set Regular` / `mode set Normal` returns to default.

### 5.3 `mode briefing "<text>"`

Calls `orchestrator.set_mission_briefing(text)`. Replaces the Godot session
objective input.

### 5.4 `mode show`

Prints the current mode + briefing + active engine_weights for that mode +
last 5 lessons generated this session (for M1-M5).

### 5.5 `mode homework`

Replaces the Godot Homework pipeline panel.

`mode homework run` invokes `dev.classifier.process(RawInput(text="/homework"))`,
then renders a 6-phase progress panel using `rich.progress.Progress`. Since the
backend is synchronous, all 6 phases run before the UI updates — so the panel
prints **after completion** as a stepper showing per-phase summary lines:

```
Phase 0 ✓  Input Assembly & Triage         12 entries → 3 batches  (deficit: logic)
Phase 1 ✓  Analysis                         5 patterns, 2 contradiction candidates
Phase 2 ✓  Processing                       2 contradictions resolved, 0 fallacies
Phase 3 ✓  Question Resolution              1 resolved, 1 new, 1 dream candidate
Phase 4 ✓  Synthesis                        3 lessons validated, 1 core update
Phase 5 ✓  Output                           overview written → reflective handoff
```

Below the stepper, print a `HomeworkRunSummary` table with all the fields
(batches_processed, lessons_validated, contradictions_resolved/unresolved,
questions_resolved/new, dream_candidates_flagged, core_memory_updates_applied,
fallacy_bias_flags count, meta_patterns count, processing_emphasis dict).

`mode homework status` prints last run summary if available.

If you later add per-phase callbacks to HomeworkPipeline, swap the post-run
print for a live updating `Progress` instance.

### 5.6 `mode reflective run` / `mode selfref <unsolved_id>`

Same pattern as homework — invoke via classifier, print result summary.

---

## 6. `dev` — Developer Workspace

Replaces Section 4 of the Godot spec.

### 6.1 `dev reward`

| Command                                                  | Replaces Godot |
|----------------------------------------------------------|----------------|
| `dev reward profiles`                                     | Profile selector dropdown |
| `dev reward profile <name>`                               | Show selected profile + sliders state |
| `dev reward override <profile> --logic 0.7 --ethics 0.3 --innovation 0.1 --attunement 0.0` | "Set as Override" button |
| `dev reward reset`                                         | "Reset to Static" button |
| `dev reward map`                                           | View MODE_TO_PROFILE table |
| `dev reward map set <mode> <profile>`                      | Edit MODE_TO_PROFILE entry |
| `dev reward learning on/off`                               | E17 Parameter Learning toggle |
| `dev reward learned`                                       | Live learned_domain_weights view |

All four domain values for `override` are required to avoid partial states.

### 6.2 `dev sleep`

Already partially covered by the top-level `sleep` namespace (§9). The `dev`
namespace just exposes the lower-level direct-control variants:

| Command                                | Replaces Godot |
|----------------------------------------|----------------|
| `dev sleep activate triage`             | "Activate Triage" button |
| `dev sleep activate rem`                 | "Activate REM" button |
| `dev sleep activate dream`               | "Activate Dream" button |
| `dev sleep exit`                          | "Exit Sleep" button |
| `dev sleep status`                        | Current sleep state + sleep metrics gauges |

The `sleep rem` / `sleep dream` top-level commands are convenience wrappers
that also auto-render the corresponding sub-workspace output (§9). Use `dev
sleep activate ...` if you want pure activation without the rich rendering.

### 6.3 `dev nt set <name> <value>` / `dev nt reset`

(Bonus — not in Godot spec.) Direct NT level injection for testing engine
behaviour at specific neurochemical states. Useful for reproducing edge cases.

### 6.4 `dev defaults`

| Command                                  | Replaces Godot |
|------------------------------------------|----------------|
| `dev defaults list`                       | Read all hardcoded defaults |
| `dev defaults get <key>`                  | Read one value |
| `dev defaults set <key> <value> [--persist]` | Edit with session-only / disk-write toggle |
| `dev defaults log`                        | Change log this session |
| `dev defaults reset`                      | Revert all session changes |

`--persist` writes to `core/hardcoded/defaults.py` (or wherever the disk
backing is); without it, only `dev.session` is mutated.

### 6.5 `dev pipeline`

| Command                          | Replaces Godot |
|----------------------------------|----------------|
| `dev pipeline last`               | Last full PipelineResult as JSON tree |
| `dev pipeline timing [--turn N]`  | Phase timing breakdown (ms) |
| `dev pipeline dispatch [--turn N]` | Engine dispatch log per engine |
| `dev pipeline errors`             | Captured PipelineValidationErrors / exceptions |

`last` should pretty-print via `rich.json.JSON`. The full PipelineResult is
huge; auto-truncate strings >500 chars unless `--full` is passed.

For phase timing to actually work, the pipeline needs to record per-phase
durations. If it doesn't already (check `core/pipeline.py`), this becomes a
**backend prerequisite** — add a `phase_timings: Dict[int, float]` field to
`PipelineState` and stamp it after each phase. Flag this in §11.

---

## 7. `atom` — Cognitools / Knowledge Map

Replaces Section 5 of the Godot spec. The hypergraph visualization is the
*one* place the Godot spec really benefits from a GUI, so the dev UI is
deliberately a **list + inspect** workflow rather than a graph rendering.

### 7.1 `atom list [--type <T>] [--filter <substring>] [--limit <N>]`

Prints all atoms in the active map as a table:

| ID  | Type        | Name        | TV (s,c)    | AV (sti,lti) |
|-----|-------------|-------------|-------------|--------------|

Filters: `--type ConceptNode/PredicateNode/...`, `--filter "name contains"`.

### 7.2 `atom show <id>`

Replaces the Godot Inspector Panel:
- Atom type badge
- Name / value
- TruthValue (strength, confidence)
- AttentionValue (STI, LTI)
- Outgoing links (ID, type, target)
- Incoming links (ID, type, source)

### 7.3 `atom add concept <name>` / `atom add predicate <name>` / etc.

One subcommand per atom type. Optional `--strength`, `--confidence`, `--sti`,
`--lti` flags.

### 7.4 `atom link <from_id> <to_id> --type <LinkType> [--strength s --confidence c]`

Adds a link between two atoms.

### 7.5 `atom set <id> --strength s` / `--confidence c` / `--sti N` / `--lti N`

Edits TruthValue / AttentionValue. Replaces the Godot inspector sliders.

### 7.6 `atom delete <id>`

Confirmation prompt before removal.

### 7.7 `atom parse "<natural language>"`

Calls the AtomSpace natural-language parser (`POST
/cognitools/atomspace/parse` in the Godot spec). The parser is in-process, so
this is just a method call on E9. Prints the resulting added atoms as a
diff table (atom id, type, name, action=ADDED/MODIFIED).

### 7.8 `atom delta`

Replaces "Changes this turn" mode. Prints atoms touched in the last turn
(added, modified, STI/LTI bumped). Uses whatever delta tracking E9 already has.

### 7.9 `atom search <substring>`

Replaces the canvas search. Returns matching atoms as `atom list` output.

### 7.10 Map management — `map ...`

| Command                          | Replaces Godot |
|----------------------------------|----------------|
| `map list`                        | Map Selector dropdown |
| `map current`                     | Show active map name |
| `map load <name>`                 | Load saved map into session |
| `map save <name>`                 | Save current session AtomSpace as named map |
| `map new <name>`                  | Start a fresh empty map |
| `map export <name> <path>`        | Export to JSON |
| `map import <path> [--name <n>]`  | Import from JSON |

### 7.11 What's deferred from Godot

- Force-directed canvas layout — out of scope.
- Node shape / size / opacity / glow rendering — out of scope.
- Live pulse animations on new HebbianLinks — out of scope.
- "Send to Conversation" right-click → use `mem ltmm unsolved push` workflow
  instead, or just `chat send "Let's talk about <name>"` manually.

---

## 8. `nt` — NT diagnostics (cross-cutting)

Top-level shortcut commands for working with the neurochemical state.
Mostly duplicates of `show neurochem` and `dev nt set`, kept short for
frequent use during testing.

| Command                | Equivalent |
|------------------------|------------|
| `nt show`               | `show neurochem` |
| `nt full`               | `show neurochem --full` |
| `nt history [N]`        | `show neurochem --history N` |
| `nt set <name> <value>` | `dev nt set <name> <value>` |
| `nt reset`              | `dev nt reset` |
| `nt metrics`            | Just the 11 NeurochemicalMetrics gauges, no NT bars |

---

## 9. `sleep` — Sleep / REM / Dream sub-workspaces

Replaces Section 6 of the Godot spec.

These commands invoke the corresponding sleep pipelines and render their
outputs as **post-completion reports**, since the underlying pipelines are
synchronous.

### 9.1 `sleep rem [--no-render]`

1. Calls `dev.classifier.process(RawInput(text="/sleep rem"))`.
2. After completion, renders the 4-phase REM panel:

   - **Phase 1 — Retroactive Learning**: list of MemoryPackets scanned, per-packet
     NT snapshot heatmap strip, detected emotional signal badge (frustration /
     curiosity / confusion / boredom / anxiety / overwhelmed), domain weight
     deltas applied table.
   - **Phase 2 — Consolidation**: MTMM packets promoted to LTMM (table),
     packets left to decay.
   - **Phase 3 — NT State**: live neurochem panel with REM prescribed baselines
     overlaid as target columns (NE-down, 5-HT-down, ACh-up, DA-up). Sigma band
     highlighted. E27 containment status.
   - **Phase 4 — Journal Write**: full journal entry prose + tags.

3. Sleep metrics panel: `consolidation_depth`, `dream_permissiveness`,
   `narrative_plasticity`.

`--no-render` skips the render and just runs the pipeline.

### 9.2 `sleep dream [--no-render]`

Same pattern.

After completion, renders:
- **Candidate queue** — three priority tiers (identity-origin, general,
  academic-origin) as separate tables.
- **Recombination output** — the novel connections generated, listed with
  their source candidate.
- **Emotional driver profile** — which signals drove the session (curiosity,
  confusion, wonder, perplexed) + domain weight adjustments applied.
- **Narrative plasticity gauge** + scene shift list (if multiple scenes).
- For each scene shift: journal entry preview.

### 9.3 `sleep triage`

Lighter — Phase 2 consolidation only. Renders the triage queue.

### 9.4 `sleep exit`

Returns to Normal mode. Equivalent to `mode set Regular`.

### 9.5 `sleep status`

Prints whether ZADOS is currently in any sleep mode + the sleep metrics
(grayed out outside sleep).

---

## 10. `sess` — Session lifecycle

| Command                              | What it does |
|--------------------------------------|--------------|
| `sess open [--previous <path>]`       | Open new session (optionally with previous SessionState) |
| `sess close`                          | Run `orchestrator.close_session()` — overview write, MTMM→LTMM consolidation, tick stagnation, end_cycle, persist cognitools |
| `sess status`                         | Current session_id, branch, turn_count, mode, briefing |
| `sess save <path>`                    | Pickle current session to disk (extension v2; not blocking v1) |
| `sess load <path>`                    | Inverse of save |
| `sess drift`                          | Run `orchestrator.check_drift()` and print divergence value |

`quit` from the REPL implies `sess close` first.

---

## 11. Backend prerequisites

Items that need to exist on the backend before the dev UI is fully wired:

1. **Per-phase timing capture** — `PipelineState.phase_timings: Dict[int,
   float]`, populated after each phase. Without this `dev pipeline timing`
   shows nothing.
2. **AtomSpace delta API** — `cognitools/atomspace/delta` equivalent. The
   Godot spec mentions it; check whether E9 already exposes a delta accessor
   and add one if not.
3. **STMM tracker exposure** — confirm `memory.stmm.tracker` is the right
   accessor for `mem stmm tracker`.
4. **Library ingest entry point** — confirm whether there's a callable that
   takes a TXT path and adds a `LibraryEntry`. If only the seed-file path
   exists, document that as the v1 ingest method.
5. **Hardcoded defaults disk write** — `defaults.py` is a Python module, so
   `dev defaults set --persist` requires a small writer that re-emits the
   module file. Decide whether to roundtrip via AST or move defaults into
   a JSON sidecar.

The point of v1 is to get a working REPL with chat + show + mem + dev
commands. The cognitools and sleep panels can land last — none of the above
prerequisites block opening a session and chatting.

---

## 12. File / module layout

Suggested layout for the dev UI itself (separate from `src/zados/`):

```
dev_ui/
├── __init__.py
├── __main__.py              # python -m dev_ui → boots ZadosShell
├── shell.py                 # ZadosShell(cmd.Cmd)
├── dev_session.py           # DevSession dataclass + bootstrap
├── status_line.py           # render_status_line(dev, console)
├── verbosity.py             # auto-display block per verbosity level
├── commands/
│   ├── __init__.py
│   ├── chat.py              # do_chat
│   ├── show.py              # do_show + engine-specific renderers
│   ├── mem.py               # do_mem (stmm/mtmm/ltmm dispatch)
│   ├── mode.py              # do_mode
│   ├── dev.py               # do_dev (reward/sleep/defaults/pipeline)
│   ├── atom.py              # do_atom + do_map
│   ├── nt.py                # do_nt
│   ├── sleep.py             # do_sleep
│   └── sess.py              # do_sess
├── render/
│   ├── __init__.py
│   ├── nt_heatmap.py        # 12-NT block-character renderer
│   ├── reward_panel.py      # show reward render
│   ├── engine_inspector.py  # E8/E18/E19/E23/E28 + JSON fallback
│   ├── memory_tables.py     # mtmm packets, ltmm lists
│   ├── sleep_panels.py      # REM/Dream/Triage post-render
│   └── homework_summary.py  # HomeworkRunSummary table
└── tests/
    └── test_shell_smoke.py
```

Keep render functions pure (input: state objects, output: `rich.Renderable`)
so they're trivially testable without a full session.

---

## 13. Build order suggestion

Iteration plan for actually coding this (not part of the spec proper, but
useful context):

1. `DevSession` bootstrap + `ZadosShell` skeleton + `chat send` + status line.
   Goal: chat with the system end-to-end, see the answer.
2. `verbosity` + auto-display block + `chat last` + `chat history`.
3. `show reward`, `show neurochem`, `show engines`, `show thinking`.
4. `mode list`, `mode set`, `mode briefing`.
5. `mem mtmm packets/context/trends`, `mem ltmm journal/unsolved`.
6. `dev reward`, `dev pipeline last/timing`.
7. `mode homework run` + `HomeworkRunSummary` render.
8. `sleep rem` + `sleep dream` post-render panels.
9. `atom list/show/add/link` + `map list/load/save`.
10. Polish: autocomplete via `prompt_toolkit`, persistent history file.

Anything past step 6 is "nice to have for v1 but ZADOS is already usable."

---

## 14. Things explicitly NOT in v1

- Force-directed graph rendering for the AtomSpace.
- Animated NT pulse strip.
- WebSocket-style streaming during a turn (synchronous only).
- Multi-session browsing.
- Personalized learning pipeline creator (Godot future feature).
- PDF/DOCX library ingestion (TXT only).
- Hardcoded defaults disk-write via AST (use JSON sidecar or skip).
- Mobile/tablet layout. (No.)

These come back when Godot returns. The Python REPL stays as the underlying
test surface either way.
