# ZADOS Developer Interface — Python REPL Specification (v1 AS-BUILT)

**Target:** Python 3.13.5 terminal-based developer interface
**Backend:** ZADOS synchronous library (in-process, no bridge server)
**Replaces (temporarily):** Godot 4.x frontend per `ZADOS_FRONTEND_SPEC.txt`
**Original draft:** 2026-05-04
**As-built revision:** 2026-05-11

> **Legend:** §-sections marked `[BUILT]` are fully implemented.
> `[PARTIAL]` means some subcommands exist, others are deferred.
> `[DEFERRED v2]` means not implemented in v1 — see §14.

---

## 0. Architecture Overview

### 0.1 No bridge server  `[BUILT]`

The Python dev UI does not use FastAPI or any bridge. All ZADOS objects
(`SessionOrchestrator`, `InputClassifier`, `MemoryLayer`, engines, etc.) are
imported directly and held in a single in-process `DevSession` object.
Synchronous calls only.

When Godot returns, FastAPI gets bolted on as a thin adapter calling into the
same backend. No work thrown away.

### 0.2 Stack  `[BUILT]`

| Concern        | Library                | Notes |
|----------------|------------------------|-------|
| REPL loop      | `cmd.Cmd` (stdlib)     | Two-tier dispatch (command + subcommand) |
| Input UX       | `prompt_toolkit`       | Installed; autocomplete deferred to v2 |
| Rendering      | `rich`                 | Panels, tables, JSON, block-char bars |
| Arg parsing    | `shlex.split` inline   | Per-handler, no argparse dependency |
| Persistence    | ZADOS stores + files   | `dev_ui_maps/*.json` for AtomSpace snapshots |

### 0.3 Single global object  `[BUILT]`

```python
@dataclass
class DevSession:
    stack:              ZadosStack
    history:            List[Any]           # all turn results (PipelineResult / dict)
    last_result:        Optional[Any]
    last_classification: Optional[Any]
    verbosity:          Literal["quiet", "normal", "nerd"]  # default: "normal"
    autoshow:           bool                # default: True
    staged_input:       Optional[str]       # pre-fills next bare-line send
    runtime_errors:     List[dict]          # caught exceptions with context

    # Convenience properties delegating into stack:
    @property orchestrator → SessionOrchestrator
    @property classifier   → InputClassifier
    @property memory       → MemoryLayer
    @property neurochem    → NeurochemicalEngine
    @property engines      → Dict[int, Any]
    @property session      → SessionState
```

`ZadosStack` is constructed by `boot_zados()` in
`src/zados/bootstrap/boot.py`. It owns the canonical engine registry and
captures per-engine boot errors in `stack.engine_errors`.

### 0.4 Layout — REPL prompt  `[BUILT]`

```
[zados] sess=a3f2…  turn=14  mode=Learning_M1  profile=curiosity_driven  verbosity=nerd
>
```

Status line prints in dim style above every prompt via `rich.Console`.  
The prompt itself is `> `. Status line redraws after every command.

### 0.5 Verbosity modes  `[BUILT]`

| Mode   | Auto-prints after each turn |
|--------|-----------------------------|
| quiet  | `final_answer` panel only |
| normal | answer + dominant emotion + directive + selected mode |
| nerd   | normal block + Phase 5 reward summary + engines run/skipped + 1-line NT delta |

### 0.6 Windows Unicode  `[BUILT]`

`__main__.py` runs `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`
at startup so block characters (▁▂▃…█) don't crash Windows cp1252 consoles.

---

## 1. Core REPL Skeleton  `[BUILT]`

### 1.1 Command groups

| Group   | Status    | Examples |
|---------|-----------|---------|
| `chat`  | BUILT     | `chat send`, `chat history`, `chat show`, `chat last`, `chat clear` |
| `show`  | BUILT     | `show reward`, `show neurochem`, `show engines`, `show engine <id>`, `show thinking`, `show classification`, `show perception` |
| `mem`   | PARTIAL   | `mem stmm`, `mem mtmm`, `mem ltmm`, `mem logs` |
| `mode`  | PARTIAL   | `mode list`, `mode set`, `mode show`, `mode briefing`, `mode homework run`, `mode reflective run` |
| `dev`   | PARTIAL   | `dev reward`, `dev nt`, `dev pipeline` |
| `atom`  | PARTIAL   | `atom list/show/search/status/add/link/set/delete` |
| `map`   | BUILT     | `map list/save/load/export/import` |
| `sleep` | BUILT     | `sleep rem/dream/triage/status/exit` |
| `nt`    | BUILT     | `nt show/full/set/reset/metrics` |
| `sess`  | PARTIAL   | `sess status/close/drift` |
| `set`   | BUILT     | `set verbosity/autoshow` |
| `quit`  | BUILT     | graceful `close_session()` + exit |

Bare lines (not starting with a known command) are routed to `chat send`.
If `dev.staged_input` is non-empty it is prepended with a newline before routing.

### 1.2 Boot sequence  `[BUILT]`

1. Parse `--no-engines` and `--log-level` flags.
2. Call `boot_zados(register_engines=not no_engines, open_session=True)`
   → returns `ZadosStack`.
   - Two-pass engine construction: zero-arg engines first, then E9
     (AtomSpace) after all ConceptNodes seeded, then E10 (PLN) and E16
     (ECAN) which depend on E9.
   - Per-engine failures are caught; stored in `stack.engine_errors`. The
     shell boots even if individual engines fail.
3. Wrap `ZadosStack` in `DevSession`.
4. Drop into `ZadosShell.cmdloop()`.

`--no-engines` skips engine construction (faster boot for memory/NT testing).

---

## 2. `chat` — Conversation Workspace  `[BUILT]`

### 2.1 `chat send <text>`

Calls `dev.classifier.process_text(text)`. The classifier handles
`/sleep rem`, `/homework`, `/reflective`, `/dream` command-prefixed input
(passed through as `chat send /sleep rem`).

After the call:
- Print turn block based on verbosity.
- Append result to `dev.history`.
- Set `dev.last_result = result`.

If the pipeline raises, error is captured into `dev.runtime_errors` and a
red panel is printed. The failed turn is **not** appended to history.

> **Change from original spec:** Classification is also captured
> (`dev.last_classification`) before `process_text` so `show classification`
> works even when the pipeline errors out.

### 2.2 `chat last`

Re-renders the most recent turn at current verbosity.

### 2.3 `chat history [N]`

Last N turns (default 10) as a `rich.Table`:

| Turn | Mode | Intent | Directive | Answer (truncated) |
|------|------|--------|-----------|---------------------|

### 2.4 `chat show <turn_idx>`

Full per-turn detail:
- `intent_archetype`, `directive`, `selected_mode`, `reward_profile_name`
- `dominant_emotion` (top-1 from E28)
- `thinking_trace` excerpt (first 200 chars)

### 2.5 `chat clear`

Clears `dev.history` and `dev.last_result`. Memory stores are untouched.

### 2.6 No streaming  `[DEFERRED v2]`

The pipeline is synchronous. Per-phase streaming requires a `phase_callback`
hook in `AnswerPipeline.process_turn()`. Deferred to v2.

---

## 3. `show` — Nerd Stats panels  `[BUILT]`

All subcommands accept optional `--turn N` (default = last turn) and
`--full` where applicable. Source is the `PipelineState` stored in
`PipelineResult.state`. If the last result was a commanded pipeline (dict),
`show` prints a hint and returns gracefully.

### 3.1 `show reward [--turn N]`  `[BUILT]`

Renders:
1. **Mode header** — panel with `mode_token` + `reward_profile_name`, border
   coloured by mode family (Normal=grey, Learning=blue, Sleep=purple,
   Dream=magenta, Homework=yellow, Reflective=cyan).
2. **Domain weights table** — `logic / ethics / innovation / human_attunement`
   with static / learned / Δ columns.
3. **Phase 5 pathway panel** — `meta_directive` (ALLOW=green, SUPPRESS=red,
   ABSTAIN=yellow) + `urgency_risk` + `selected_mode` + tonic/phasic flags.
4. **Domain scores panel** — per-domain result scores (if present).
5. **NT signals panel** — `nt_signals` dict; positive=green, negative=red.

### 3.2 `show neurochem [--turn N] [--full]`  `[BUILT]`

Three panels:
- **Neurotransmitters** — 12 NTs as block-character bars in canonical order:
  `glu gaba da 5ht ne ach oxt mor cb1 crh cortisol histamine`.
  `--full` adds numeric values (3 dp).
- **Oscillations** — 6 bands (delta/theta/alpha/beta/gamma/sigma) + 3
  cross-band signals (theta_gamma / alpha_beta / delta_sigma).
- **Metrics** — 11 `NeurochemicalMetrics` in 2-column gauge table. Sleep
  metrics (dream_permissiveness, consolidation_depth, narrative_plasticity)
  are greyed out when not in a sleep mode.

`show neurochem --history N`  `[DEFERRED v2]` — NT sparkline timeline.

### 3.3 `show engines [--turn N]`  `[BUILT]`

32-engine grid (6 columns, 6 rows).
- **bold green** — ran this turn
- **yellow** — skipped
- **dim** — registered but idle
- **red dim** — not registered (boot failure or `--no-engines`)

Below: cluster-level `engine_weights` sorted by weight descending.

> **Correction from spec:** The spec said "29-engine grid" but v1 includes
> E31 and E32 (ReflLearn and ReflIdent), making it 32 slots.

### 3.4 `show engine <id> [--turn N]`  `[BUILT]`

Structured views for:
- **E8** — ranked facets table (name, score)
- **E18** — entity triples table (subject, relation, object)
- **E19** — patterns table (id/name, kind, confidence)
- **E23** — intent archetype + confidence bar + intent vector coefficients
- **E28** — 28-emotion intensity table, sorted by intensity

All others: rich JSON tree (strings truncated at 500 chars; `--full` lifts
the limit).

### 3.5 `show thinking [--turn N]`  `[BUILT]`

Prints `thinking_trace` in italic dim inside a "thinking" panel.
If the thinking phase was skipped, shows `SKIPPED (reason)` in the title.

`thinking save/list/show/delete`  `[DEFERRED v2]`

### 3.6 `show classification [<text>]`  `[BUILT — bonus, not in original spec]`

With no args: renders the last captured `InputClassification`.  
With text: classifies the text ad-hoc and renders the result immediately.

Fields: `input_type`, `sub_type`, `variant`, `route_target`, `confidence`,
`learning_mode_number` (if any), `raw_text` (first 120 chars).

### 3.7 `show perception [--turn N]`  `[BUILT]`

Pretty-prints `PerceptionSnapshot`: `intent_archetype`, `intent_confidence`,
`intent_vector` (top 15, sorted), `ranked_facets` (top 10), `entity_triples`
(top 10), `pattern_list` (top 10).

---

## 4. `mem` — Memory Workspace  `[PARTIAL]`

Four tiers: `stmm`, `mtmm`, `ltmm`, `logs`.

### 4.1 `mem stmm`  `[BUILT]`

```
mem stmm                → same as mem stmm current
mem stmm current        → labelled field list from dev.memory.stmm
mem stmm tracker        → STMM access tracker stats
```

Read-only. STMM resets when the next turn starts.

### 4.2 `mem mtmm`  `[BUILT]`

```
mem mtmm                → same as mem mtmm packets
mem mtmm packets [N]    → last N packets (default 10) as a table
mem mtmm packet <id>    → single packet detail + NT heatmap + reward scores
mem mtmm trends         → trend summary panel
```

Table columns: turn #, time, intent, emotion, trust, sig, user msg (40c),
response (40c).

`mem mtmm context get/set/reset`  `[DEFERRED v2]`  
`mem mtmm search`  `[DEFERRED v2]`

### 4.3 `mem ltmm`  `[PARTIAL]`

The LTMM command uses **dotted `namespace.store` notation** rather than deep
subcommand trees:

```
mem ltmm                                → overview of all 17 stores
mem ltmm list                           → all stores with record counts
mem ltmm <namespace>.<store>            → same as ...list
mem ltmm <namespace>.<store> list [N]   → first N records (default 20)
mem ltmm <namespace>.<store> show <id> → single record detail
```

**Available stores** (use `mem ltmm list` to see counts):

| Dotted name | Store | Namespace |
|---|---|---|
| `knowledge.lessons` | LessonsStore | knowledge |
| `knowledge.library` | LibraryStore | knowledge |
| `knowledge.maps` | KnowledgeMapStore | knowledge |
| `knowledge.buffer` | AcademicBuffer | knowledge |
| `knowledge.cognitools` | CognitoolsDataStore | knowledge |
| `journal.entries` | JournalStore | journal |
| `thoughts.blocks` | HeldThinkingBlocksStore | thoughts |
| `thoughts.overview` | OverviewLogStore | thoughts |
| `thoughts.questions` | GeneralQuestionsStore | thoughts |
| `identity.core` | CoreMemoryStore | identity |
| `identity.development` | DevelopmentLogStore | identity |
| `identity.conclusions` | ConclusionsStore | identity |
| `identity.journal` | IdentityJournalStore | identity |
| `unsolved.buffer` | UnsolvedBuffer | unsolved |
| `unsolved.academic` | AcademicQuestionsStore | unsolved |
| `session.pending_updates` | PendingUpdateQueue | session |
| `session.domain_weights` | DomainWeightStore | session |

Deeper LTMM subcommands from the original spec (library ingest, notebook
editor, unsolved push/selfref/tag, journal trigger) are `[DEFERRED v2]`.

### 4.4 `mem logs`  `[BUILT — not in original spec]`

Eight specialised log stores surfaced as a 4th memory tier:

```
mem logs                    → overview of all 8 logs
mem logs <name> [N]         → last N entries from that log
```

Log names: `learning`, `dream`, `rem`, `homework`, `reflective`,
`contradiction`, `bias`, `fallacy`.

---

## 5. `mode` — Learning Workspace  `[PARTIAL]`

### 5.1 `mode list`  `[BUILT]`

Table of all 25 mode tokens in 5 categories:
Regular | Learning (M1–M5) | Sleep (S1–S4) | Dialectic (D1–D2) | Self-Referential (SR1).

### 5.2 `mode set <token>`  `[BUILT]`

Supported forms:
- `M1` … `M5` — sets `active_learning_mode`, `session_mode=learning`, resolves reward profile via `profile_for_learning_mode(n)`.
- `regular` / `normal` — resets to default mode and default profile.
- Any other token — stored as `initial_mode` and profile resolved via `profile_for_mode(token)`.

### 5.3 `mode briefing [<text>]`  `[BUILT]`

With text: calls `orchestrator.set_mission_briefing(text)`.  
With no args: prints the current briefing.

### 5.4 `mode show`  `[BUILT]`

Current mode + briefing + active engine_weights for that mode.

### 5.5 `mode homework run`  `[BUILT]`

Invokes `/homework` via `classifier.process_text()`, then renders a
6-phase stepper + `HomeworkRunSummary` table.

`mode homework status`  `[DEFERRED v2]`

### 5.6 `mode reflective run`  `[BUILT]`

Invokes `/reflective` via `classifier.process_text()`, then renders
the E31 meta-learning + E32 identity-coherence summary panels.

`mode selfref <unsolved_id>`  `[DEFERRED v2]`

---

## 6. `dev` — Developer Workspace  `[PARTIAL]`

### 6.1 `dev reward`  `[PARTIAL]`

```
dev reward profiles                                  → list all profiles
dev reward profile <name>                            → full profile detail
dev reward map                                       → MODE_TO_PROFILE table
dev reward learned                                   → live learned_domain_weights delta view
dev reward override --logic V --ethics V --innovation V --attunement V
dev reward reset                                     → revert overrides
```

> **Note:** All four weights are required for `override` to avoid partial
> states. Partial override is rejected with a usage error.

`dev reward learning on/off`  `[DEFERRED v2]`  
`dev reward map set <mode> <profile>`  `[DEFERRED v2]`

### 6.2 `dev nt`  `[BUILT]`

```
dev nt show [--full]          → NT state table
dev nt metrics                → NeurochemicalMetrics only (no bars)
dev nt set <name> <value>     → inject NT level (float 0–1)
dev nt reset                  → restore homeostatic baseline
```

NT name aliases: `da`/`dopamine`, `5ht`/`5-ht`/`serotonin`, `ne`/`norepinephrine`,
`ach`/`acetylcholine`, `gaba`, `oxt`/`oxytocin`, `cb1`/`endocannabinoid`,
`cortisol`, `crh`, `glu`, `mor`, `histamine`.

### 6.3 `dev pipeline`  `[BUILT]`

```
dev pipeline last [--full] [--turn N]     → PipelineResult as rich JSON tree
dev pipeline dispatch [--turn N]          → engine dispatch log
dev pipeline errors                       → runtime_errors list
dev pipeline errors <idx>                 → error #idx with full traceback
```

`dev pipeline timing`  `[DEFERRED v2]` — requires `phase_timings` field in
`PipelineState` (see §11.1).

### 6.4 `dev defaults`  `[DEFERRED v2]`

Stub present — prints "not implemented in v1".

### 6.5 `dev sleep activate`  `[DEFERRED v2]`

Use top-level `sleep rem` / `sleep dream` / `sleep exit` instead.

---

## 7. `atom` — AtomSpace  `[PARTIAL]`

E9 (AtomSpace) must be registered for atom commands to work. If E9 is missing,
each command prints an error and exits gracefully.

### 7.1 `atom list [--type T] [--name SUBSTR] [N]`  `[BUILT]`

Table: ID, type, name, TV (strength, confidence), AV (STI, LTI).  
Default limit: 30. N overrides the limit.

### 7.2 `atom show <id_or_name>`  `[BUILT]`

Accepts atom UUID or atom name.  
Fields: type badge, name/value, TruthValue, AttentionValue,
outgoing links (id, type, target), incoming links (id, type, source).

### 7.3 `atom add node <Type> <name> [--strength S --confidence C]`  `[BUILT]`

> **Change from original spec:** A single `atom add node` subcommand handles
> all node types (the type is the first positional arg after `node`).
> Separate `atom add predicate` etc. are not needed.

### 7.4 `atom link <Type> <id1> <id2> [...] [--strength S --confidence C]`  `[BUILT]`

> **Change from spec:** Syntax is `atom link <Type> <ids...>` (type first),
> matching the positional ordering of AtomSpace's `add_link()`. Compared to
> spec's proposed `atom link <from> <to> --type <Type>`.

### 7.5 `atom set <id_or_name> [--strength S --confidence C --sti N --lti N]`  `[BUILT]`

Accepts name or UUID. Edits TruthValue and/or AttentionValue.

### 7.6 `atom delete <id_or_name>`  `[BUILT]`

Resolves name → UUID before calling `remove_atom()`. Verifies deletion and
confirms to user.

> **Bug fixed:** Original implementation called `remove_atom(name)` instead
> of resolving to the actual UUID first. Fixed.

### 7.7 `atom search <substring> [N]`  `[BUILT]`

Returns matching atoms as an `atom list`-style table. Default limit: 30.

### 7.8 `atom status`  `[BUILT]`

AtomSpace statistics: total atoms, per-type counts, attention-focus size.

### 7.9 `atom parse "<natural language>"`  `[DEFERRED v2]`

### 7.10 `atom delta`  `[DEFERRED v2]`

---

## 8. `map` — AtomSpace snapshot persistence  `[BUILT]`

> **Architecture change from original spec:** The spec proposed using
> `KnowledgeMapStore` for persistence. In v1, maps use file-based persistence
> in `ROOT/dev_ui_maps/<name>.json` via `AtomSpace.export_to_dict()` /
> `import_from_dict()`. This avoids schema mismatch between `KnowledgeMap`
> dataclass expectations and raw AtomSpace exports.
>
> `ROOT/dev_ui_maps/` is gitignored.

```
map list                  → saved snapshots in dev_ui_maps/
map save <name> [desc]    → export current AtomSpace to dev_ui_maps/<name>.json
map load <name>           → import snapshot (merges into current AtomSpace)
map export <path>         → copy snapshot to arbitrary file path
map import <path>         → import from arbitrary file path
```

A full post-bootstrap snapshot is ~4700 atoms / ~2 MB.

`map current` / `map new`  `[DEFERRED v2]`

---

## 9. `nt` — NT diagnostics shortcut  `[BUILT]`

| Command | Equivalent |
|---------|------------|
| `nt` (bare) | `nt show` |
| `nt show` | `dev nt show` |
| `nt full` | `dev nt show --full` |
| `nt set <name> <value>` | `dev nt set <name> <value>` |
| `nt reset` | `dev nt reset` |
| `nt metrics` | `dev nt metrics` |

`nt history [N]`  `[DEFERRED v2]`

---

## 10. `sleep` — Sleep / REM / Dream pipelines  `[BUILT]`

All sleep commands invoke the corresponding classifier command and render a
structured post-completion summary (synchronous pipelines; no live progress).

### 10.1 `sleep rem`

Calls `classifier.process_text("/sleep rem")`.  
Renders `render_rem_result(result)`:
- Status + session_id + processing_time_s
- Packets scanned / consolidated
- Dominant emotional signals table (name, strength)
- Domain weight adjustments table

### 10.2 `sleep dream`

Calls `classifier.process_text("/sleep dream")`.  
Renders `render_dream_result(result)`:
- Status + stats (candidates found/processed, novel connections)
- Emotional driver profile table
- Domain weight adjustments

### 10.3 `sleep triage`

No dedicated triage pipeline. Falls back to `sleep rem` with a note.

### 10.4 `sleep status`

Renders a panel: `session_mode` (AWAKE / IN SLEEP), `active_learning_mode`,
`initial_mode`, `reward_profile`, `branch`.

### 10.5 `sleep exit`

Sets `session.session_mode = "regular"`, clears `active_learning_mode`.

`sleep homework` / `sleep reflective` are exposed via `mode homework run` and
`mode reflective run`.

---

## 11. `sess` — Session lifecycle  `[PARTIAL]`

```
sess status     → session_id, branch, modes, profile, briefing, turn count, engine counts
sess close      → orchestrator.close_session() + prints summary dict
sess drift      → orchestrator.check_drift() + prints divergence value
```

`sess open` / `sess save` / `sess load`  `[DEFERRED v2]`

---

## 12. `set` — Shell settings  `[BUILT]`

```
set verbosity quiet|normal|nerd
set autoshow on|off
```

---

## 13. Backend prerequisites

### 13.1 Per-phase timing  `[DEFERRED]`

`dev pipeline timing` requires `PipelineState.phase_timings: Dict[int, float]`
stamped after each phase in `AnswerPipeline`. Not yet added; the command
returns nothing without it.

### 13.2 AtomSpace delta API  `[DEFERRED]`

`atom delta` requires E9 to expose a delta accessor (atoms touched this turn).
Not yet implemented.

### 13.3 STMM tracker exposure  `[BUILT]`

`memory.stmm` exposes a `tracker` attribute — confirmed working.

### 13.4 Library ingest entry point  `[DEFERRED]`

`mem ltmm knowledge.library ingest <path>` requires a callable that accepts a
TXT path and adds a `LibraryEntry`. Only seed-file ingestion exists in v1.

### 13.5 Hardcoded defaults disk write  `[DEFERRED]`

`dev defaults set --persist` would require re-emitting a Python module via
AST or a JSON sidecar. Not in v1 scope.

---

## 14. Things NOT in v1 (deferred to v2)

All of these come back when Godot returns or as the need arises:

| Feature | Notes |
|---|---|
| Force-directed AtomSpace graph | GUI-only — Godot |
| Animated NT pulse strip | GUI-only |
| Per-phase streaming during turns | Needs `phase_callback` hook in pipeline |
| NT sparkline history (`show neurochem --history`) | Minor — add when useful |
| `thinking save/list/show/delete` | `mem ltmm thoughts.blocks` works as a workaround |
| `mem ltmm knowledge.library ingest` | TXT path ingest endpoint not yet exposed |
| `mem ltmm knowledge.notebook add` | No `$EDITOR` integration |
| `mem ltmm unsolved push/selfref/tag` | Use `staged_input` workaround or manual `chat send` |
| `mem ltmm journal trigger` | Direct `JournalTool` invocation |
| `dev reward learning on/off` | E17 toggle not yet exposed |
| `dev reward map set <mode> <profile>` | MODE_TO_PROFILE edit |
| `dev defaults list/get/set/reset` | Stub present, not implemented |
| `dev sleep activate` | Use top-level `sleep rem/dream` instead |
| `atom parse "<text>"` | AtomSpace NLP parser |
| `atom delta` | Atoms-touched-this-turn diff |
| `map new` / `map current` | Minor UX |
| `sess open --previous` / `sess save` / `sess load` | Session pickle serialization |
| `sess open` (new session mid-run) | Lifecycle management |
| `mode selfref <unsolved_id>` | Self-referential pipeline routing |
| `mode homework status` | Last-run summary cache |
| `sleep triage` dedicated pipeline | Falls back to REM |
| `prompt_toolkit` autocomplete | Installed but not wired to REPL |
| Multi-session browsing | Out of scope |
| PDF/DOCX library ingestion | TXT only for v1 |
| Mobile layout | No |

---

## 15. File / module layout  (actual v1)

```
ROOT/
├── src/zados/
│   ├── bootstrap/
│   │   ├── __init__.py          ← exports ZadosStack, boot_zados
│   │   └── boot.py              ← boot_zados(), ZadosStack dataclass
│   └── ...                      ← all other ZADOS backend modules
│
├── dev_ui/
│   ├── __init__.py              ← sys.path bootstrap (src/ added here)
│   ├── __main__.py              ← entry: python -m dev_ui [--no-engines] [--log-level L]
│   ├── dev_session.py           ← DevSession dataclass + record_error()
│   ├── shell.py                 ← ZadosShell(cmd.Cmd) — all do_* handlers
│   ├── render.py                ← unwrap_pipeline_result(), render_status_line(),
│   │                                render_turn_block(), render_history_table(),
│   │                                render_answer_panel(), render_turn_detail()
│   ├── render_show.py           ← show reward/neurochem/engines/engine/thinking/
│   │                                classification/perception renderers
│   ├── render_mem.py            ← STMM / MTMM / LTMM / logs renderers
│   ├── render_mode.py           ← mode list + mode inspector
│   ├── render_dev.py            ← reward profiles, NT tooling, pipeline debug,
│   │                                apply_nt_set/reset, apply_reward_override/reset
│   ├── render_commanded.py      ← REM / Dream / Homework / Reflective renderers +
│   │                                render_sleep_status()
│   └── render_atom.py           ← atom list/show/search/status + CRUD +
│                                   map file-based persistence
│
├── dev_ui_maps/                 ← AtomSpace JSON snapshots (gitignored)
├── tests/
├── .gitignore
└── ZADOS_DEV_UI.md              ← usage guide (quickstart + git setup)
```

> **Change from original spec:** The spec proposed `commands/` and `render/`
> subdirectories. The v1 implementation uses a flat `dev_ui/` layout with
> one `render_<group>.py` per command group. This is simpler to navigate
> and avoids extra `__init__.py` files.

---

## 16. Build order (as executed)

1. `DevSession` + `ZadosStack` + `ZadosShell` skeleton + `chat send` + status line → end-to-end chat working.
2. Verbosity auto-display + `chat last` / `chat history`.
3. `show reward`, `show neurochem`, `show engines`, `show engine <id>`, `show thinking`, `show perception`.
4. `mode list`, `mode set`, `mode briefing`, `mode show`.
5. `mem stmm`, `mem mtmm`, `mem ltmm` (dotted notation), `mem logs`.
6. `dev reward`, `dev nt`, `dev pipeline` + `apply_*` mutation helpers.
7. `mode homework run` + `mode reflective run` → HomeworkRunSummary + Reflective renderers.
8. `sleep rem` + `sleep dream` + `sleep status` + `sleep exit`.
9. `atom list/show/search/status/add/link/set/delete` + `map list/save/load/export/import`.
10. `nt` shortcuts, `set`, `sess status/close/drift`, `quit`.

Steps 1–10 are all complete. Everything in §14 is v2.
