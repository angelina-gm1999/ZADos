# ZADOS Developer UI — Usage Guide

A terminal-based Python REPL for interacting with the ZADOS backend without a full frontend. Covers all 10 command groups, testing workflows, and git setup.

---

## Quick Start

```bash
cd ROOT
python -m dev_ui
```

**Flags:**

| Flag | Effect |
|---|---|
| `--no-engines` | Boot without cognitive engines (faster, for memory/NT testing) |
| `--log-level DEBUG` | Verbose boot output |

At the prompt you can type **bare text to chat**, or use any command below.

```
zados> hello, how are you?
zados> teach me about thermodynamics
zados> chat send what is a paradox?
```

The shell routes bare lines directly to `chat send`. All commands are case-insensitive.

---

## Command Reference

### `chat` — Send input & inspect history

```
chat send <text>          Send text to ZADOS (same as typing bare text)
chat history              Last 10 turns (turn#, mode, NT snapshot, reply preview)
chat history --n 20       Last 20 turns
chat show                 Full detail of last turn
chat show --turn 3        Full detail of turn #3
chat clear                Wipe turn history
```

**Tips:**
- `chat show` is your main debug view — shows AI reply, classification, engine results, NT state
- After a learning-mode turn (ZADOS says "I'll teach you about X") use `chat show` to see the full `LearningModeResult` detail

---

### `show` — Inspect last pipeline state

```
show nt                   Neurochemical heatmap (12 NTs, block-char bars)
show nt full              All NT fields including raw values
show reward               Domain weights — static / learned / delta + phase 5 dispatch
show engines              6×6 color-coded engine grid (green=ran, yellow=skipped, dim=idle)
show engines <id>         Deep inspector for a specific engine (1-30)
show thinking             Thinking/reasoning trace from last turn
show perception           Perception state (entity extraction, salience, etc.)
show class                Full InputClassification dump
```

**Engine colors:**
- **Green** — ran this turn
- **Yellow** — skipped (insufficient confidence or NT gating)
- **Dim** — idle / not registered

---

### `mem` — Memory layers

```
mem stmm                  Current STMM buffer (recency-sorted packets)
mem stmm tracker          STMM access tracker stats
mem mtmm                  MTMM packet list (consolidated episodes)
mem mtmm <id>             Single MTMM packet detail
mem mtmm trends           MTMM pattern trends
mem ltmm                  LTMM store overview (17 named stores)
mem ltmm list             All stores with record counts
mem ltmm show <store>     Records from a specific store (see store names below)
mem logs                  All 8 specialized log stores overview
mem logs <log_name>       Entries from a specific log
```

**LTMM store names** (partial list — use `mem ltmm list` for full list):
`lessons`, `knowledge_maps`, `conclusions`, `identity_journal`, `questions`, `unsolved`, `cognitools`, `learning_logs`, `domain_weights`, `reward_history`

**Log names:**
`learning`, `dream`, `rem`, `homework`, `reflective`, `contradiction`, `bias`, `fallacy`

---

### `mode` — Session / learning modes

```
mode list                 All 25 mode tokens in 5 categories
mode show                 Current active mode + briefing + cluster weights
mode set <token>          Switch to a mode (e.g. M1, S3, D2, SL1, REG)
```

**Mode categories:**

| Prefix | Category | Examples |
|---|---|---|
| `REG` | Regular | Default conversation |
| `M` | Learning | M1 Teach, M2 Socratic, M3 Research … |
| `S` | Sleep | S1 REM, S2 Dream, S3 Homework, S4 Reflective |
| `D` | Dialectic | D1 Opposition, D2 Debate |
| `SR` | Self-referential | SR1 Introspection |

---

### `dev` — Developer / reward tooling

```
dev reward profiles               List all reward profiles
dev reward profile <name>         Full profile detail (domain weights, phase 5 dispatch)
dev reward map                    Domain weight map (all domains)
dev reward learned                Learned weight deltas since session open
dev reward set <domain> <value>   Override a domain weight (float 0-1)
dev reward reset                  Reset all overrides to profile defaults

dev nt state                      NT state table (same as `show nt full`)
dev nt metrics                    NT homeostasis metrics only
dev nt set <nt> <value>           Manually set an NT level (float 0-1)
dev nt reset                      Reset all NTs to homeostatic baseline

dev pipeline last                 Last pipeline dispatch summary
dev pipeline dispatch             All pipeline routes registered
dev pipeline errors               Runtime error list (collected since boot)
dev pipeline errors <n>           Error #n detail + traceback
```

---

### `nt` — Neurochemistry shortcuts

```
nt                        Same as `show nt` — compact heatmap
nt full                   Full NT table
nt set <nt> <value>       Same as `dev nt set`
nt reset                  Same as `dev nt reset`
```

**NT name aliases** (any of these work):
- `da` / `DA` / `dopamine`
- `5ht` / `5HT` / `5-ht` / `serotonin`
- `ne` / `NE` / `norepinephrine`
- `ach` / `ACh` / `acetylcholine`
- `gaba` / `GABA`
- `oxt` / `OXT` / `oxytocin`
- `cb1` / `CB1` / `endocannabinoid`
- `cortisol` / `Cortisol`

---

### `sleep` — Commanded pipelines (REM / Dream / Homework / Reflective)

```
sleep status              Current session_mode (awake / IN SLEEP)
sleep rem                 Run REM consolidation pipeline
sleep dream               Run Dream / creative recombination pipeline
sleep homework            Run Homework 6-phase structured processing
sleep reflective          Run Reflective meta-learning + identity coherence
```

These pipelines run immediately in the background and render a structured summary when done. They don't require switching to sleep mode first — useful for forcing a processing cycle mid-session during testing.

---

### `atom` — AtomSpace (Engine 9)

```
atom list                 First 50 atoms (id, type, name, sti)
atom list --n 100         First 100 atoms
atom show <id_or_name>    Single atom detail (truth value, attention, outgoing links)
atom search <query>       Search atoms by name prefix / substring
atom status               AtomSpace stats (total atoms, type counts, AF size)

atom add node <type> <name>          Add a ConceptNode (or any node type)
atom add link <type> <src> <dst>     Add a link between two atoms by name
atom set <id_or_name> sti <value>    Set STI on an atom
atom delete <id_or_name>             Remove an atom
```

**Common atom types:** `ConceptNode`, `PredicateNode`, `NumberNode`, `ListLink`, `EvaluationLink`, `InheritanceLink`, `SimilarityLink`

---

### `map` — AtomSpace snapshot persistence

```
map list                  Saved snapshots in ROOT/dev_ui_maps/
map save <name>           Export current AtomSpace to dev_ui_maps/<name>.json
map load <name>           Import a snapshot (merges into current AtomSpace)
map export <name> <path>  Copy snapshot to an arbitrary file path
map import <path> <name>  Import from an arbitrary file path into map library
```

Snapshots are plain JSON files — open them in VS Code to inspect the full atom graph. A full post-bootstrap snapshot is ~4700 atoms / ~2 MB.

---

### `sess` — Session lifecycle

```
sess info                 Session ID, mode, branch, uptime
sess close                Run close_session() — overview write → consolidate → persist
sess open                 Open a new session (replaces current)
```

---

### `set` — Shell settings

```
set verbosity quiet       Minimal output (reply only)
set verbosity normal      Reply + key stats (default)
set verbosity nerd        Full structured detail block after every turn
set autoshow on/off       Toggle automatic `show` panel after each turn
```

---

### `quit` / `exit` / `Ctrl-C`

Gracefully closes the session (runs `close_session()`) then exits.

---

## Testing Workflows

### 1 — Basic conversation loop
```
zados> hello
zados> chat show                  # inspect classification, engines, NT state
zados> show engines               # see which of the 30 engines fired
zados> show engines 28            # E28 Emotional Detection deep view
```

### 2 — Force a learning mode
```
zados> teach me about Bayesian inference
zados> chat show                  # LearningModeResult + stage breakdown
zados> mem ltmm show lessons      # new lesson should be persisted
zados> show reward                # check if domain weights shifted
```

### 3 — Neurochemistry experiment
```
zados> nt set da 0.9              # high dopamine
zados> teach me something new     # should affect novelty-seeking engines
zados> show engines 17            # E17 Reward-Based Learning — note learning_rate
zados> nt reset
```

### 4 — Memory inspection cycle
```
zados> hello
zados> teach me about entropy
zados> mem stmm                   # current STMM buffer
zados> mem mtmm                   # any consolidated episodes?
zados> mem ltmm list              # all 17 stores with counts
zados> mem ltmm show lessons
```

### 5 — Run sleep pipelines
```
zados> sleep rem
zados> sleep homework             # 6-phase — shows per-phase breakdown
zados> sleep reflective           # E31 meta-learning + E32 identity coherence
zados> mem logs rem               # check REM log entries
```

### 6 — AtomSpace inspection
```
zados> atom status                # total atoms + type distribution
zados> atom search emotion        # find emotion-related nodes
zados> atom show <id>             # inspect a specific atom
zados> map save baseline          # snapshot before experiments
zados> atom add node ConceptNode my_test_concept
zados> atom search my_test
zados> atom delete my_test_concept
zados> map load baseline          # restore if needed
```

### 7 — Error investigation
```
zados> dev pipeline errors        # list any caught runtime errors
zados> dev pipeline errors 0      # full traceback for error #0
```

---

## Running Tests

From `ROOT/`:

```bash
python -m pytest tests/neurochem/ tests/reward/ tests/memory/ tests/cog_engines/ tests/core/ -v
```

For a specific subsystem:

```bash
python -m pytest tests/cog_engines/ -v -k "atomspace"
python -m pytest tests/memory/ -v
python -m pytest tests/neurochem/ -v --tb=short
```

---

## Git Setup

The repo root is `ROOT/`. A `.gitignore` is already present. Steps to verify and do an initial commit:

```bash
cd ROOT

# Check current status
git status

# If not already initialised
git init
```

### Recommended `.gitignore` additions

Add these to `ROOT/.gitignore` if not already present:

```
# Dev UI runtime artifacts
dev_ui_maps/
*.jsonl

# Codebase snapshots (regenerated)
ZADOS_FULL_CODEBASE.txt

# Logs / temp
*.log
*.tmp
```

### Initial commit (if starting fresh)

```bash
git add .
git commit -m "Initial ZADOS commit — backend + dev UI"
```

### Typical development commit flow

```bash
# After adding a feature or fixing a bug:
git add ROOT/src/zados/...          # be specific — avoid git add .
git add ROOT/tests/...
git commit -m "feat(engine): add E22 contextual learning + 85 tests"

# After a dev UI change:
git add ROOT/dev_ui/
git commit -m "dev_ui: add atom deep inspector + map round-trip"
```

### Connecting to GitHub

```bash
# Create the repo on github.com first, then:
git remote add origin https://github.com/<you>/ZADOS.git
git branch -M main
git push -u origin main
```

### Useful git aliases for this project

```bash
git config alias.st   "status --short"
git config alias.lg   "log --oneline --graph --decorate -20"
git config alias.dt   "diff HEAD"
```

---

## File Map

```
ROOT/
├── src/zados/              — All ZADOS backend source
│   ├── bootstrap/          — boot_zados(), ZadosStack
│   ├── cognitive_engines/  — 29 engines (py_engines/ + cognitools/)
│   ├── memory/             — STMM / MTMM / LTMM stores
│   ├── neurochem/          — 12-NT neurochemical layer
│   ├── reward/             — Reward profiles + domain weights
│   └── ...
├── dev_ui/                 — This developer REPL
│   ├── __init__.py         — sys.path bootstrap
│   ├── __main__.py         — Entry point (python -m dev_ui)
│   ├── dev_session.py      — DevSession dataclass
│   ├── shell.py            — cmd.Cmd shell, all do_* handlers
│   ├── render.py           — Core renderers + unwrap_pipeline_result()
│   ├── render_show.py      — NT / reward / engine / perception views
│   ├── render_mem.py       — STMM / MTMM / LTMM / log views
│   ├── render_mode.py      — Mode list + mode inspector
│   ├── render_dev.py       — Reward profiles + NT tooling + pipeline debug
│   ├── render_commanded.py — REM / Dream / Homework / Reflective renderers
│   └── render_atom.py      — AtomSpace + map persistence
├── dev_ui_maps/            — Saved AtomSpace snapshots (gitignored)
├── tests/                  — pytest suites
├── .gitignore
└── ZADOS_DEV_UI.md         — This file
```
