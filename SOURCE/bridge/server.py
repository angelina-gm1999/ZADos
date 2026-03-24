"""
ZADOS Bridge Server — FastAPI application.

Exposes ZADOS to the Godot frontend over HTTP REST + WebSocket.

Start with:
    cd ROOT
    .venv/Scripts/python.exe -m uvicorn bridge.server:app --reload --port 8000

Endpoints
---------
POST /session/open          Open a new session; returns session_id + branch
POST /session/briefing      Set the mission briefing for the active session
GET  /session/state         Full session state snapshot
GET  /metrics               Live neurochem metrics (poll)

POST /process               Synchronous turn; returns full result
WS   /stream/process        Streaming turn (phase events + pseudo-token stream)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bridge.bootstrap import ZADOSStack, build_stack
from bridge.serializers import (
    pipeline_state_snapshot,
    process_response,
    safe_json,
    session_snapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global stack (built once at startup)
# ---------------------------------------------------------------------------

_stack: ZADOSStack | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stack
    _stack = build_stack()
    yield
    log.info("ZADOS bridge shutting down.")


app = FastAPI(title="ZADOS Bridge", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # localhost only in practice; tighten if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_stack() -> ZADOSStack:
    if _stack is None:
        raise HTTPException(status_code=503, detail="ZADOS stack not initialised.")
    return _stack


def _require_session(stack: ZADOSStack) -> Any:
    session = stack.orchestrator.session
    if session is None:
        raise HTTPException(status_code=400, detail="No active session. POST /session/open first.")
    return session


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class OpenSessionRequest(BaseModel):
    previous_session_id: str = ""   # reserved for future cross-session handoff


class BriefingRequest(BaseModel):
    briefing: str


class ProcessRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post("/session/open")
async def open_session(req: OpenSessionRequest = OpenSessionRequest()):
    """Open a new ZADOS session."""
    stack = _require_stack()
    session = stack.orchestrator.open_session()
    log.info("Session opened: %s (branch %s)", session.session_id, session.branch)
    return {
        "session_id": session.session_id,
        "branch":     session.branch,
        "initial_mode": session.initial_mode,
    }


@app.post("/session/briefing")
async def set_briefing(req: BriefingRequest):
    """Set the mission briefing for the active session."""
    stack = _require_stack()
    _require_session(stack)
    stack.orchestrator.set_mission_briefing(req.briefing)
    return {"status": "ok"}


@app.get("/session/state")
async def get_session_state():
    """Return a full snapshot of the current session state."""
    stack = _require_stack()
    session = _require_session(stack)
    return session_snapshot(session)


@app.get("/metrics")
async def get_metrics():
    """Return live neurochem metrics (poll this between turns for live strip)."""
    stack = _require_stack()
    try:
        metrics = stack.neurochem.get_neurosymbolic_readout()
        if hasattr(metrics, "as_dict"):
            metrics = metrics.as_dict()
        return safe_json(metrics)
    except Exception as exc:
        log.exception("Metrics readout failed.")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Synchronous process endpoint
# ---------------------------------------------------------------------------

@app.post("/process")
async def process_turn(req: ProcessRequest):
    """Run one full turn synchronously and return the complete result."""
    stack = _require_stack()
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _run_turn(stack, req.text),
        )
    except Exception as exc:
        log.exception("process_turn failed.")
        raise HTTPException(status_code=500, detail=str(exc))

    session = stack.orchestrator.session
    return process_response(result, session)


# ---------------------------------------------------------------------------
# Streaming WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/stream/process")
async def stream_process(ws: WebSocket):
    """
    Streaming turn over WebSocket.

    Client sends:   {"text": "Hello!"}

    Server sends a sequence of JSON messages:
      {"type": "phase_complete", "phase": 1, "data": {...}}
      {"type": "phase_complete", "phase": 2, "data": {...}}
      {"type": "phase_complete", "phase": 3, "data": {...}}
      {"type": "token",          "phase": 4, "text": "..."}   (chunks)
      {"type": "phase_complete", "phase": 4, "data": {...}}
      {"type": "phase_complete", "phase": 5, "data": {...}}
      {"type": "token",          "phase": 6, "text": "..."}   (chunks)
      {"type": "complete",       "result": {...}}

    Note: the pipeline runs synchronously in a thread; phase events are
    emitted after the pipeline completes.  Token chunks are the thinking
    trace and final answer streamed character-by-character for the UI.
    Real per-phase streaming requires instrumentation inside AnswerPipeline
    and is a future enhancement.
    """
    await ws.accept()
    stack = _require_stack()

    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()

    try:
        data = await ws.receive_json()
    except WebSocketDisconnect:
        return

    text = data.get("text", "")
    if not text:
        await ws.send_json({"type": "error", "message": "Empty text."})
        await ws.close()
        return

    # Run the pipeline in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _run_turn(stack, text),
        )
    except Exception as exc:
        log.exception("stream_process pipeline failed.")
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close()
        return

    # Extract PipelineResult from wrapper types (LearningModeResult, SelfRefResult)
    pipeline_result = _unwrap_pipeline_result(result)
    state = getattr(pipeline_result, "state", None) if pipeline_result else None

    # --- Emit phase_complete events for phases 1-3 ---
    if state is not None:
        phase_data = [
            (1, safe_json(getattr(state, "perception", None))),
            (2, safe_json(getattr(state, "modulation", None))),
            (3, safe_json(getattr(state, "dispatch",   None))),
        ]
        for phase_num, data_payload in phase_data:
            await ws.send_json({
                "type":  "phase_complete",
                "phase": phase_num,
                "data":  data_payload or {},
            })

    # --- Phase 4: stream thinking trace ---
    thinking_text = ""
    if state is not None and getattr(state, "thinking", None):
        thinking_text = getattr(state.thinking, "thinking_trace", "") or ""

    if thinking_text:
        for chunk in _text_chunks(thinking_text, size=10):
            await ws.send_json({"type": "token", "phase": 4, "text": chunk})
            await asyncio.sleep(0.008)

    await ws.send_json({
        "type":  "phase_complete",
        "phase": 4,
        "data":  safe_json(getattr(state, "thinking", None)) if state else {},
    })

    # --- Phase 5 ---
    await ws.send_json({
        "type":  "phase_complete",
        "phase": 5,
        "data":  safe_json(getattr(state, "reward", None)) if state else {},
    })

    # --- Phase 6: stream final answer ---
    final_answer = ""
    if pipeline_result is not None:
        final_answer = getattr(pipeline_result, "final_answer", "") or ""

    if final_answer:
        for chunk in _text_chunks(final_answer, size=10):
            await ws.send_json({"type": "token", "phase": 6, "text": chunk})
            await asyncio.sleep(0.008)

    await ws.send_json({
        "type":  "phase_complete",
        "phase": 6,
        "data":  safe_json(getattr(state, "answer", None)) if state else {},
    })

    # --- Complete ---
    session = stack.orchestrator.session
    await ws.send_json({
        "type":   "complete",
        "result": process_response(result, session),
    })

    await ws.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Last pipeline result — stored so /dev/* endpoints can inspect it.
_last_result: Any = None


def _run_turn(stack: ZADOSStack, text: str) -> Any:
    """Run one turn through the InputClassifier and increment turn_count."""
    global _last_result
    from zados.core.types import RawInput
    result = stack.classifier.process(RawInput(text=text))
    _last_result = result
    # Increment turn counter on the session
    if stack.orchestrator.session is not None:
        stack.orchestrator.session.turn_count += 1
    return result


def _unwrap_pipeline_result(result: Any) -> Any:
    """Extract a PipelineResult from wrapper types."""
    if result is None:
        return None
    # PipelineResult has final_answer directly
    if hasattr(result, "final_answer"):
        return result
    # LearningModeResult / SelfRefResult wrap it in .pipeline_result
    inner = getattr(result, "pipeline_result", None)
    if inner is not None:
        return inner
    return None


def _text_chunks(text: str, size: int = 10):
    """Yield text in chunks of `size` characters."""
    for i in range(0, len(text), size):
        yield text[i: i + size]


# ---------------------------------------------------------------------------
# Session — set mode
# ---------------------------------------------------------------------------

class SetModeRequest(BaseModel):
    mode: str


# Maps display name → (session_mode, active_learning_mode)
_MODE_MAP: dict = {
    "Normal":         ("regular",  None),
    "M1":             ("learning", "M1"),
    "M2":             ("learning", "M2"),
    "M3":             ("learning", "M3"),
    "M4":             ("learning", "M4"),
    "M5":             ("learning", "M5"),
    "Homework":       ("meta",     None),
    "Reflective":     ("meta",     None),
    "SelfReflective": ("regular",  None),   # auto-activated by input markers
}


@app.post("/session/set_mode")
async def set_session_mode(req: SetModeRequest):
    """Change the active mode on the current session (sets all three fields)."""
    stack = _require_stack()
    session = _require_session(stack)
    session.initial_mode = req.mode
    session_mode, learning_mode = _MODE_MAP.get(req.mode, ("regular", None))
    session.session_mode         = session_mode
    session.active_learning_mode = learning_mode
    log.info("Mode → %s  (session_mode=%s  learning=%s)",
             req.mode, session_mode, learning_mode)
    return {
        "active_mode":          req.mode,
        "session_mode":         session_mode,
        "active_learning_mode": learning_mode,
        "status":               "ok",
    }


# ---------------------------------------------------------------------------
# Learning pipeline endpoints
# ---------------------------------------------------------------------------

@app.post("/homework")
async def run_homework():
    """Trigger HomeworkPipeline via /homework command text."""
    stack = _require_stack()
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: _run_turn(stack, "/homework")
        )
    except Exception as exc:
        log.exception("Homework pipeline failed.")
        raise HTTPException(status_code=500, detail=str(exc))
    return safe_json(result) or {"status": "completed"}


@app.post("/reflective")
async def run_reflective():
    """Trigger ReflectivePipeline via /reflective command text."""
    stack = _require_stack()
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: _run_turn(stack, "/reflective")
        )
    except Exception as exc:
        log.exception("Reflective pipeline failed.")
        raise HTTPException(status_code=500, detail=str(exc))
    return safe_json(result) or {"status": "completed"}


# ---------------------------------------------------------------------------
# Memory — helpers
# ---------------------------------------------------------------------------

def _items(items: list) -> dict:
    """Wrap a list in {items, count} so all endpoints return dicts."""
    return {"items": safe_json(items), "count": len(items)}


def _stmm_snapshot(stmm: Any) -> dict:
    """Compact snapshot of the current STMMStore for the Memory workspace."""
    d: dict = {}

    buf = getattr(stmm, "active_message_buffer", None)
    if buf:
        msgs = getattr(buf, "messages", [])
        d["message_count"] = len(msgs)
        for m in reversed(msgs):
            sp = str(getattr(m, "speaker", ""))
            if "USER" in sp.upper():
                d["latest_user"] = (getattr(m, "text", "") or "")[:400]
                break

    intention = getattr(stmm, "intention_analysis", None)
    if intention:
        d["intent_archetype"]    = getattr(intention, "primary_archetype",   "")
        d["primary_intention"]   = getattr(intention, "primary_intention",   "")
        d["confidence"]          = float(getattr(intention, "confidence",     0.0))

    reflection = getattr(stmm, "cortical_reflection", None)
    if reflection:
        d["active_mode"]         = getattr(reflection, "active_mode",                   "Normal")
        d["identity_coherence"]  = getattr(reflection, "identity_coherence_status",     "")
        d["processing_anomalies"]= safe_json(getattr(reflection, "processing_anomalies", []))
        vr = getattr(reflection, "verbal_reflection", "") or ""
        d["verbal_reflection"]   = vr[:500]

    emotions = getattr(stmm, "emotion_detection", None)
    if emotions:
        d["tone_valence"]   = float(getattr(emotions, "tone_valence",   0.0))
        d["tone_warmth"]    = float(getattr(emotions, "tone_warmth",    0.0))
        d["tone_coherence"] = float(getattr(emotions, "tone_coherence", 0.0))
        d["user_emotions"]  = safe_json(getattr(emotions, "user_emotion_signals", {}))

    tracker = getattr(stmm, "brain_process_tracker", None)
    if tracker:
        d["stage_flags"]       = safe_json(getattr(tracker, "pipeline_stage_flags", {}))
        executions             = getattr(tracker, "executions", [])
        d["engines_run_count"] = len([e for e in executions
                                      if not getattr(e, "skipped", True)])

    return d


def _packet_summary(p: Any) -> dict:
    return {
        "packet_id":             getattr(p, "packet_id",             ""),
        "timestamp":             str(getattr(p, "timestamp",         "")),
        "turn_index":            getattr(p, "turn_index",            0),
        "user_message":          (getattr(p, "user_message",         "") or "")[:200],
        "system_response":       (getattr(p, "system_response",      "") or "")[:200],
        "intention":             getattr(p, "intention",             ""),
        "verbal_summary":        getattr(p, "verbal_summary",        ""),
        "verbal_emotion_labels": safe_json(getattr(p, "verbal_emotion_labels", [])),
        "emotion_vector":        safe_json(getattr(p, "emotion_vector",       {})),
        "neurochemical_snapshot":safe_json(getattr(p, "neurochemical_snapshot",{})),
        "reward_scores":         safe_json(getattr(p, "reward_scores",        {})),
        "flags":                 safe_json(getattr(p, "flags",                [])),
        "trust_weight":          float(getattr(p, "trust_weight",          0.0)),
        "emotional_significance":float(getattr(p, "emotional_significance", 0.0)),
    }


# ---------------------------------------------------------------------------
# Memory — STMM
# ---------------------------------------------------------------------------

@app.get("/memory/stmm")
async def get_stmm():
    stack = _require_stack()
    _require_session(stack)
    return _stmm_snapshot(stack.memory.stmm)


# ---------------------------------------------------------------------------
# Memory — MTMM
# ---------------------------------------------------------------------------

@app.get("/memory/mtmm/context")
async def get_mtmm_context():
    stack = _require_stack()
    _require_session(stack)
    mtmm = stack.memory.mtmm
    packets = mtmm.get_all_packets()
    trends = safe_json(getattr(mtmm, "trends", {}))
    return {
        "packet_count":      len(packets),
        "recent_intentions": [getattr(p, "intention", "") for p in packets[-5:]],
        "trends":            trends,
    }


@app.get("/memory/mtmm/packets")
async def get_mtmm_packets():
    stack = _require_stack()
    _require_session(stack)
    packets = stack.memory.mtmm.get_all_packets()
    return _items([_packet_summary(p) for p in packets])


@app.get("/memory/mtmm/trends")
async def get_mtmm_trends():
    stack = _require_stack()
    _require_session(stack)
    trends = getattr(stack.memory.mtmm, "trends", None)
    return safe_json(trends) or {}


# ---------------------------------------------------------------------------
# Memory — LTMM Journal
# ---------------------------------------------------------------------------

class TriggerJournalRequest(BaseModel):
    trigger_source: str = "dev_interface"
    notes: list = []


@app.post("/memory/ltmm/journal/trigger")
async def trigger_journal_entry(req: TriggerJournalRequest = TriggerJournalRequest()):
    """Manually trigger a journal entry via the DEV trigger type."""
    stack = _require_stack()
    session = _require_session(stack)
    try:
        from zados.memory.long_term.journal.entry import JournalEntry, JournalTrigger
        entry = JournalEntry(
            trigger=JournalTrigger.DEV,
            trigger_source=req.trigger_source,
            session_id=session.session_id,
            pipeline_notes=req.notes,
        )
        stack.memory.journal_store.write(entry)
        return {"status": "ok", "entry_id": entry.entry_id}
    except Exception as exc:
        log.exception("Failed to trigger journal entry.")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/memory/ltmm/journal")
async def get_journal():
    stack = _require_stack()
    _require_session(stack)
    entries = stack.memory.journal_store.get_recent(20)
    return _items([{
        "entry_id":          getattr(e, "entry_id",    ""),
        "timestamp":         str(getattr(e, "timestamp", "")),
        "trigger":           safe_json(getattr(e, "trigger", "")),
        "prose":             (getattr(e, "prose", "") or "")[:600],
        "reflection_prompts":safe_json(getattr(e, "reflection_prompts", [])),
        "emotion_snapshot":  safe_json(getattr(e, "emotion_snapshot",  {})),
        "nt_snapshot":       safe_json(getattr(e, "nt_snapshot",       {})),
        "tags":              safe_json(getattr(e, "tags",              [])),
        "review_status":     safe_json(getattr(e, "review_status",    "")),
    } for e in entries])


# ---------------------------------------------------------------------------
# Memory — LTMM Thoughts
# ---------------------------------------------------------------------------

class CreateHeldBlockRequest(BaseModel):
    thought_fragment: str
    emotion_tag: str = ""
    context_summary: str = ""
    pipeline_phase: str = ""


@app.post("/memory/ltmm/thoughts/held_blocks")
async def create_held_block(req: CreateHeldBlockRequest):
    """Save a thinking trace as a held block in LTMM."""
    stack = _require_stack()
    _require_session(stack)
    try:
        from zados.memory.long_term.thoughts.types import HeldThinkingBlock
        block = HeldThinkingBlock(
            thought_fragment=req.thought_fragment,
            emotion_tag=req.emotion_tag,
            context_summary=req.context_summary,
            pipeline_phase=req.pipeline_phase,
            session_id=stack.orchestrator.session.session_id,
        )
        stack.memory.thoughts.held_blocks.write(block)
        return {"status": "ok", "block_id": block.block_id}
    except Exception as exc:
        log.exception("Failed to save held block.")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/memory/ltmm/thoughts/held_blocks")
async def get_held_blocks():
    stack = _require_stack()
    _require_session(stack)
    blocks = stack.memory.thoughts.held_blocks.get_all()
    return _items([{
        "block_id":        getattr(b, "block_id",        ""),
        "timestamp":       str(getattr(b, "timestamp",   "")),
        "thought_fragment":(getattr(b, "thought_fragment","") or "")[:400],
        "emotion_tag":     getattr(b, "emotion_tag",     ""),
        "context_summary": getattr(b, "context_summary", ""),
        "pipeline_phase":  getattr(b, "pipeline_phase",  ""),
        "tags":            safe_json(getattr(b, "tags",  [])),
        "reviewed":        getattr(b, "reviewed",        False),
    } for b in blocks])


@app.get("/memory/ltmm/thoughts/overview_logs")
async def get_overview_logs():
    stack = _require_stack()
    _require_session(stack)
    logs = stack.memory.thoughts.overview_logs.get_all()
    return _items([{
        "log_id":          getattr(l, "log_id",          ""),
        "timestamp":       str(getattr(l, "timestamp",   "")),
        "session_id":      getattr(l, "session_id",      ""),
        "summary":         (getattr(l, "summary", "") or "")[:500],
        "mode_sequence":   safe_json(getattr(l, "mode_sequence",   [])),
        "subject_tags":    safe_json(getattr(l, "subject_tags",    [])),
        "dominant_emotions":safe_json(getattr(l, "dominant_emotions",[])),
        "open_threads":    safe_json(getattr(l, "open_threads",    [])),
    } for l in logs])


@app.get("/memory/ltmm/thoughts/general_questions")
async def get_general_questions():
    stack = _require_stack()
    _require_session(stack)
    questions = stack.memory.thoughts.general_questions.get_all()
    return _items([{
        "question_id":   getattr(q, "question_id", ""),
        "formulation":   getattr(q, "formulation", ""),
        "source":        getattr(q, "source",      ""),
        "priority":      float(getattr(q, "priority", 0.0)),
        "stagnation_count": getattr(q, "stagnation_count", 0),
        "resolved":      getattr(q, "resolved",    False),
        "tags":          safe_json(getattr(q, "tags", [])),
    } for q in questions])


# ---------------------------------------------------------------------------
# Memory — LTMM Knowledge
# ---------------------------------------------------------------------------

@app.get("/memory/ltmm/knowledge/lessons")
async def get_lessons():
    stack = _require_stack()
    _require_session(stack)
    lessons = stack.memory.knowledge.lessons.get_all()
    return _items([{
        "lesson_id":        getattr(l, "lesson_id",        ""),
        "content":          (getattr(l, "content", "") or "")[:400],
        "subject_category": getattr(l, "subject_category", ""),
        "source_mode":      getattr(l, "source_mode",      ""),
        "confidence":       float(getattr(l, "confidence", 0.0)),
        "validation_status":getattr(l, "validation_status",""),
        "tags":             safe_json(getattr(l, "tags",   [])),
        "created_at":       str(getattr(l, "created_at",  "")),
        "reinforcement_count": getattr(l, "reinforcement_count", 0),
    } for l in lessons])


@app.get("/memory/ltmm/knowledge/notebook")
async def get_notebook():
    stack = _require_stack()
    _require_session(stack)
    notes = stack.memory.knowledge.notebook.get_all()
    return _items([{
        "note_id":          getattr(n, "note_id",          ""),
        "content":          (getattr(n, "content", "") or "")[:400],
        "subject_category": getattr(n, "subject_category", ""),
        "source_mode":      getattr(n, "source_mode",      ""),
        "tags":             safe_json(getattr(n, "tags",   [])),
        "timestamp":        str(getattr(n, "timestamp",    "")),
    } for n in notes])


@app.get("/memory/ltmm/knowledge/academic_buffer")
async def get_academic_buffer():
    stack = _require_stack()
    _require_session(stack)
    entries = stack.memory.knowledge.academic_buffer.get_all()
    return _items([{
        "entry_id":           getattr(e, "entry_id",           ""),
        "concept_formulation":(getattr(e, "concept_formulation","") or "")[:400],
        "subject_category":   getattr(e, "subject_category",   ""),
        "source_engine":      getattr(e, "source_engine",      ""),
        "blocking_reason":    getattr(e, "blocking_reason",    ""),
        "stagnation_cycles":  getattr(e, "stagnation_cycles",  0),
        "resolved":           getattr(e, "resolved",           False),
        "resolution_note":    getattr(e, "resolution_note",    ""),
        "timestamp":          str(getattr(e, "timestamp",      "")),
        "dream_candidate":    getattr(e, "is_dream_candidate",
                              lambda threshold=5: getattr(e, "stagnation_cycles", 0) >= threshold)(),
    } for e in entries])


class ResolveRequest(BaseModel):
    note: str = ""


@app.post("/memory/ltmm/knowledge/academic_buffer/{entry_id}/resolve")
async def resolve_academic_entry(entry_id: str, req: ResolveRequest = ResolveRequest()):
    stack = _require_stack()
    _require_session(stack)
    try:
        stack.memory.knowledge.academic_buffer.resolve(entry_id, req.note)
        return {"status": "ok", "entry_id": entry_id}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Memory — LTMM Library
# ---------------------------------------------------------------------------

@app.get("/memory/ltmm/knowledge/library")
async def get_library():
    """List all library entries (books, articles, documents)."""
    stack = _require_stack()
    _require_session(stack)
    entries = stack.memory.knowledge.library.get_all()
    return _items([{
        "entry_id":     getattr(e, "entry_id",     ""),
        "title":        getattr(e, "title",        ""),
        "content":      (getattr(e, "content", "") or "")[:400],
        "source_type":  getattr(e, "source_type",  ""),
        "domain":       getattr(e, "domain",       ""),
        "tags":         safe_json(getattr(e, "tags", [])),
        "timestamp":    str(getattr(e, "timestamp", "")),
    } for e in entries])


@app.get("/memory/ltmm/knowledge/library/search")
async def search_library(q: str = "", limit: int = 10):
    """Search the library by keyword."""
    stack = _require_stack()
    _require_session(stack)
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")
    results = stack.memory.knowledge.library.search(q, limit=limit)
    return _items([{
        "score":        round(score, 4),
        "entry_id":     getattr(e, "entry_id",     ""),
        "title":        getattr(e, "title",        ""),
        "content":      (getattr(e, "content", "") or "")[:400],
        "source_type":  getattr(e, "source_type",  ""),
        "domain":       getattr(e, "domain",       ""),
        "tags":         safe_json(getattr(e, "tags", [])),
    } for score, e in results])


class LibraryIngestRequest(BaseModel):
    title: str
    content: str
    source_type: str = "document"
    domain: str = ""
    tags: list = []
    strategy: str = "auto"


@app.post("/memory/ltmm/knowledge/library/ingest")
async def ingest_library_text(req: LibraryIngestRequest):
    """Ingest raw text into the library (with optional chunking)."""
    stack = _require_stack()
    _require_session(stack)
    from zados.memory.long_term.knowledge.library.importer import import_text
    result = import_text(
        store=stack.memory.knowledge.library,
        title=req.title,
        content=req.content,
        domain=req.domain,
        tags=req.tags,
        source_type=req.source_type,
        strategy=req.strategy,
    )
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return {
        "status": "ok",
        "title": result.title,
        "strategy": result.strategy,
        "entries_created": result.entries_created,
        "total_chars": result.total_chars,
        "group_id": result.group_id,
        "entry_ids": result.entry_ids,
    }


class LibraryImportFileRequest(BaseModel):
    file_path: str
    title: str = ""
    source_type: str = "book"
    domain: str = ""
    tags: list = []
    strategy: str = "auto"


@app.post("/memory/ltmm/knowledge/library/import")
async def import_library_file(req: LibraryImportFileRequest):
    """Import a .txt file from disk into the library."""
    stack = _require_stack()
    _require_session(stack)
    from zados.memory.long_term.knowledge.library.importer import import_file
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: import_file(
            store=stack.memory.knowledge.library,
            file_path=req.file_path,
            title=req.title,
            domain=req.domain,
            tags=req.tags,
            source_type=req.source_type,
            strategy=req.strategy,
        ),
    )
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return {
        "status": "ok",
        "title": result.title,
        "strategy": result.strategy,
        "entries_created": result.entries_created,
        "total_chars": result.total_chars,
        "group_id": result.group_id,
        "entry_ids": result.entry_ids,
    }


# ---------------------------------------------------------------------------
# Memory — LTMM Identity
# ---------------------------------------------------------------------------

@app.get("/memory/ltmm/identity/core")
async def get_core_memories():
    stack = _require_stack()
    _require_session(stack)
    memories = stack.memory.identity.core.get_all()
    return _items([{
        "memory_id":    getattr(m, "memory_id",    ""),
        "content":      (getattr(m, "content", "") or "")[:500],
        "memory_type":  getattr(m, "memory_type",  ""),
        "tags":         safe_json(getattr(m, "tags", [])),
        "created_at":   str(getattr(m, "created_at", "")),
        "version":      getattr(m, "version", 1),
    } for m in memories])


@app.get("/memory/ltmm/identity/hardcoded")
async def get_hardcoded():
    stack = _require_stack()
    _require_session(stack)
    entries = stack.memory.identity.hardcoded.get_all()
    return _items([{
        "entry_id": getattr(e, "entry_id", ""),
        "content":  (getattr(e, "content", "") or "")[:500],
        "category": getattr(e, "category", ""),
        "tags":     safe_json(getattr(e, "tags", [])),
    } for e in entries])


# ---------------------------------------------------------------------------
# Memory — LTMM Identity — Development (Conclusions + Identity Journal)
# ---------------------------------------------------------------------------

@app.get("/memory/ltmm/identity/development")
async def get_identity_development():
    """Return identity conclusions + identity journal entries (development log)."""
    stack = _require_stack()
    _require_session(stack)
    conclusions = []
    journal_entries = []
    conclusions_store = getattr(stack.memory.identity, "conclusions", None)
    if conclusions_store:
        try:
            conclusions = [{
                "conclusion_id":      getattr(c, "conclusion_id",      ""),
                "content":            (getattr(c, "content", "") or "")[:500],
                "conclusion_type":    getattr(c, "conclusion_type",    ""),
                "confidence":         float(getattr(c, "confidence",   0.5)),
                "tags":               safe_json(getattr(c, "tags",     [])),
                "created_at":         str(getattr(c, "created_at",     "")),
                "reinforcement_count":getattr(c, "reinforcement_count",0),
            } for c in conclusions_store.get_all()]
        except Exception:
            pass
    journal_store = getattr(stack.memory.identity, "journal", None)
    if journal_store:
        try:
            journal_entries = [{
                "entry_id":        getattr(e, "entry_id",        ""),
                "entry_type":      safe_json(getattr(e, "entry_type", "")),
                "content":         (getattr(e, "content", "") or "")[:500],
                "source_pipeline": getattr(e, "source_pipeline", ""),
                "emotion_tags":    safe_json(getattr(e, "emotion_tags", [])),
                "tags":            safe_json(getattr(e, "tags",          [])),
                "timestamp":       str(getattr(e, "timestamp",          "")),
            } for e in journal_store.get_all()]
        except Exception:
            pass
    return {"conclusions": conclusions, "journal_entries": journal_entries}


@app.get("/memory/ltmm/identity/alignment")
async def get_identity_alignment():
    """Run the alignment checker against the current STMM context."""
    stack = _require_stack()
    _require_session(stack)
    try:
        from zados.memory.long_term.identity.alignment import IdentityAlignmentChecker
        checker = IdentityAlignmentChecker(stack.memory.identity.hardcoded)
        result = checker.check(stack.memory.stmm)
        return {
            "axiom_notes":        safe_json(getattr(result, "axiom_notes",        [])),
            "value_notes":        safe_json(getattr(result, "value_notes",        [])),
            "constraint_notes":   safe_json(getattr(result, "constraint_notes",   [])),
            "personality_prompts":safe_json(getattr(result, "personality_prompts", [])),
            "flags":              safe_json(getattr(result, "flags",              [])),
        }
    except Exception as exc:
        log.exception("Alignment check failed.")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Memory — LTMM Unsolved
# ---------------------------------------------------------------------------

@app.get("/memory/ltmm/unsolved")
async def get_unsolved():
    stack = _require_stack()
    _require_session(stack)
    entries = stack.memory.thoughts.unsolved_buffer.get_all_active()
    return _items([{
        "entry_id":            getattr(e, "entry_id",            ""),
        "concept_formulation": (getattr(e, "concept_formulation","") or "")[:400],
        "source_engine":       getattr(e, "source_engine",       ""),
        "blocking_reason":     getattr(e, "blocking_reason",     ""),
        "stagnation_cycles":   getattr(e, "stagnation_cycles",   0),
        "resolution_attempts": safe_json(getattr(e, "resolution_attempts", [])),
        "evidence_accumulated":safe_json(getattr(e, "evidence_accumulated",[])),
        "resolved":            getattr(e, "resolved",            False),
        "timestamp":           str(getattr(e, "timestamp",       "")),
        "dream_candidate":     getattr(e, "is_dream_candidate",
                               lambda threshold=5: getattr(e, "stagnation_cycles", 0) >= threshold)(),
    } for e in entries])


@app.post("/memory/ltmm/unsolved/{entry_id}/resolve")
async def resolve_unsolved(entry_id: str, req: ResolveRequest = ResolveRequest()):
    stack = _require_stack()
    _require_session(stack)
    try:
        stack.memory.thoughts.unsolved_buffer.resolve(entry_id, req.note)
        return {"status": "ok", "entry_id": entry_id}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Dev workspace endpoints
# ---------------------------------------------------------------------------

def _get_nt_value(val: Any) -> float:
    """Extract a float from various NT readout shapes."""
    if isinstance(val, dict):
        for k in ("tonic", "level", "value", "concentration"):
            if k in val:
                try:
                    return float(val[k])
                except (TypeError, ValueError):
                    pass
        return 0.5
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.5


def _estimate_sleep_phase(readout: dict) -> str:
    """Derive a sleep phase label from NT concentrations."""
    ach  = _get_nt_value(readout.get("ACh",  readout.get("ach",  0.5)))
    ne   = _get_nt_value(readout.get("NE",   readout.get("ne",   0.5)))
    gaba = _get_nt_value(readout.get("GABA", readout.get("gaba", 0.5)))
    if ach > 0.75 and ne < 0.15:
        return "DREAM"
    if gaba > 0.70 and ne < 0.30:
        return "REM_PROCESSING"
    if gaba > 0.55 and ne < 0.45:
        return "TRIAGE"
    return "WAKING"


@app.get("/dev/neurochem")
async def get_dev_neurochem():
    """Full neurosymbolic readout + estimated sleep phase."""
    stack = _require_stack()
    try:
        readout = stack.neurochem.get_neurosymbolic_readout()
        if hasattr(readout, "as_dict"):
            readout = readout.as_dict()
        data = safe_json(readout) or {}
        data["_sleep_phase"] = _estimate_sleep_phase(data)
        return data
    except Exception as exc:
        log.exception("Dev neurochem readout failed.")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/reward")
async def get_dev_reward():
    """Phase5 reward results from the last turn."""
    stack = _require_stack()
    _require_session(stack)
    if _last_result is None:
        return {"status": "no_result_yet", "domains": {}, "meta_directive": {}, "nt_signals": {}}

    pipeline_result = _unwrap_pipeline_result(_last_result)
    state  = getattr(pipeline_result, "state", None) if pipeline_result else None
    phase5 = getattr(state, "reward", None) if state else None

    if phase5 is None:
        return {"status": "no_reward_data", "domains": {}, "meta_directive": {}, "nt_signals": {}}

    domains: dict = {}
    for domain, result in (getattr(phase5, "domain_results", {}) or {}).items():
        subscores = {}
        for k, v in (getattr(result, "subscores", {}) or {}).items():
            score = getattr(v, "score", v)
            try:
                subscores[str(k)] = float(score)
            except (TypeError, ValueError):
                subscores[str(k)] = 0.0
        domains[str(domain)] = {
            "general_score": float(getattr(result, "general_score", 0.0)),
            "subscores":     subscores,
        }

    meta = getattr(phase5, "meta_directive", None)
    meta_dict: dict = {}
    if meta is not None:
        meta_dict = {
            "allow_output": bool(getattr(meta, "allow_output", True)),
            "abstain":      bool(getattr(meta, "abstain",      False)),
            "suppress":     bool(getattr(meta, "suppress",     False)),
            "directives":   safe_json(getattr(meta, "directives", {})) or {},
            "flags":        safe_json(getattr(meta, "flags",       {})) or {},
        }

    return {
        "status":         "ok",
        "selected_mode":  str(getattr(phase5, "selected_mode",  "")),
        "urgency_risk":   float(getattr(phase5, "urgency_risk",  0.0)),
        "domains":        domains,
        "meta_directive": meta_dict,
        "nt_signals":     safe_json(getattr(phase5, "nt_signals", {})) or {},
    }


class OverrideWeightsRequest(BaseModel):
    logic_weight: float = 0.25
    ethics_weight: float = 0.25
    innovation_weight: float = 0.25
    attunement_weight: float = 0.25


@app.post("/dev/reward/override_weights")
async def override_reward_weights(req: OverrideWeightsRequest):
    """Override learned domain weights on the current session."""
    stack = _require_stack()
    session = _require_session(stack)
    session.learned_domain_weights = {
        "logic_weight":      req.logic_weight,
        "ethics_weight":     req.ethics_weight,
        "innovation_weight": req.innovation_weight,
        "attunement_weight": req.attunement_weight,
    }
    log.info("Reward weights overridden: %s", session.learned_domain_weights)
    return {"status": "ok", "weights": session.learned_domain_weights}


@app.post("/dev/reward/reset_weights")
async def reset_reward_weights():
    """Reset learned domain weights to empty (uses static profile)."""
    stack = _require_stack()
    session = _require_session(stack)
    session.learned_domain_weights = {}
    log.info("Reward weights reset to static profile.")
    return {"status": "ok", "weights": {}}


@app.get("/dev/pipeline")
async def get_dev_pipeline():
    """pipeline_state_snapshot from the last turn."""
    stack = _require_stack()
    _require_session(stack)
    if _last_result is None:
        return {"status": "no_result_yet"}
    pipeline_result = _unwrap_pipeline_result(_last_result)
    state = getattr(pipeline_result, "state", None) if pipeline_result else None
    if state is None:
        return {"status": "no_state"}
    try:
        return pipeline_state_snapshot(state)
    except Exception as exc:
        log.exception("Dev pipeline snapshot failed.")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/dev/sleep/trigger")
async def trigger_sleep_cycle():
    """Manually trigger a sleep cycle via the /sleep command."""
    stack = _require_stack()
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: _run_turn(stack, "/sleep")
        )
    except Exception as exc:
        log.exception("Sleep trigger failed.")
        raise HTTPException(status_code=500, detail=str(exc))
    return safe_json(result) or {"status": "triggered"}


@app.post("/dev/sleep/rem")
async def trigger_rem():
    """Trigger REM pipeline specifically; returns structured REMResult."""
    stack = _require_stack()
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: _run_turn(stack, "/sleep rem")
        )
    except Exception as exc:
        log.exception("REM pipeline failed.")
        raise HTTPException(status_code=500, detail=str(exc))
    return safe_json(result) or {"status": "completed"}


@app.post("/dev/sleep/dream")
async def trigger_dream():
    """Trigger Dream pipeline specifically; returns structured DreamResult."""
    stack = _require_stack()
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: _run_turn(stack, "/sleep dream")
        )
    except Exception as exc:
        log.exception("Dream pipeline failed.")
        raise HTTPException(status_code=500, detail=str(exc))
    return safe_json(result) or {"status": "completed"}


@app.get("/dev/sleep/state")
async def get_sleep_state():
    """Return current sleep-relevant state: NT concentrations, unsolved candidates, MTMM packet info."""
    stack = _require_stack()
    _require_session(stack)
    # NT readout for sleep phase estimation
    try:
        readout = stack.neurochem.get_neurosymbolic_readout()
        if hasattr(readout, "as_dict"):
            readout = readout.as_dict()
        nt_data = safe_json(readout) or {}
    except Exception:
        nt_data = {}

    # Unsolved buffer candidates
    try:
        unsolved = stack.memory.thoughts.unsolved_buffer.get_all_active()
        dream_candidates = [
            {
                "entry_id":            getattr(e, "entry_id", ""),
                "concept_formulation": (getattr(e, "concept_formulation", "") or "")[:200],
                "stagnation_cycles":   getattr(e, "stagnation_cycles", 0),
                "source_engine":       getattr(e, "source_engine", ""),
                "dream_candidate":     getattr(e, "is_dream_candidate",
                                       lambda threshold=5: getattr(e, "stagnation_cycles", 0) >= threshold)(),
            }
            for e in unsolved
            if getattr(e, "is_dream_candidate",
                       lambda threshold=5: getattr(e, "stagnation_cycles", 0) >= threshold)()
        ]
    except Exception:
        dream_candidates = []

    # MTMM packet summary for consolidation preview
    try:
        packets = stack.memory.mtmm.get_all_packets()
        packet_summary = {
            "total": len(packets),
            "high_significance": len([p for p in packets
                                      if float(getattr(p, "emotional_significance", 0)) > 0.6]),
            "low_trust": len([p for p in packets
                              if float(getattr(p, "trust_weight", 1.0)) < 0.3]),
        }
    except Exception:
        packet_summary = {"total": 0, "high_significance": 0, "low_trust": 0}

    return {
        "sleep_phase": _estimate_sleep_phase(nt_data),
        "nt_snapshot": nt_data,
        "dream_candidates": dream_candidates,
        "dream_candidate_count": len(dream_candidates),
        "mtmm_packets": packet_summary,
    }


# ---------------------------------------------------------------------------
# Plumbing diagnostics
# ---------------------------------------------------------------------------

@app.post("/dev/plumbing")
async def run_plumbing_tests():
    """Run server-side plumbing diagnostics and return a structured report.

    Tests pipeline phase data flow, memory tier read/write, neurochemical
    modulation influence, and consolidation pathways.  No LLM calls needed.
    """
    stack = _require_stack()
    loop = asyncio.get_event_loop()
    try:
        from bridge.plumbing import run_all
        report = await loop.run_in_executor(None, lambda: run_all(stack))
        return report
    except Exception as exc:
        log.exception("Plumbing tests crashed.")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Map workspace endpoints
# ---------------------------------------------------------------------------

def _find_atomspace_engine(stack: ZADOSStack) -> Any:
    """Walk known stack paths to find the AtomSpaceEngine (has get_all_atoms)."""
    candidates: list = []
    for top_attr in ("classifier", "orchestrator"):
        top = getattr(stack, top_attr, None)
        if top is None:
            continue
        for sub_attr in ("engines", "_engines", "py_engines", "_py_engines",
                         "answer_pipeline", "_pipeline"):
            sub = getattr(top, sub_attr, None)
            if sub is None:
                continue
            if isinstance(sub, dict):
                candidates.extend(sub.values())
            elif isinstance(sub, (list, tuple)):
                candidates.extend(sub)
            else:
                inner = getattr(sub, "engines", None) or getattr(sub, "_engines", None)
                if inner:
                    candidates.extend(inner.values() if isinstance(inner, dict) else inner)
    for e in candidates:
        if hasattr(e, "get_all_atoms"):
            return e
    return None


def _atom_to_node(atom: Any) -> dict:
    tv = getattr(atom, "truth_value", None)
    av = getattr(atom, "attention_value", None)
    return {
        "id":            getattr(atom, "atom_id",      ""),
        "label":         getattr(atom, "name",         "") or "",
        "type":          str(getattr(atom, "atom_type", "")),
        "strength":      float(getattr(tv, "strength",   0.5)) if tv else 0.5,
        "confidence":    float(getattr(tv, "confidence", 0.0)) if tv else 0.0,
        "sti":           float(getattr(av, "sti",        0.0)) if av else 0.0,
        "lti":           float(getattr(av, "lti",        0.0)) if av else 0.0,
        "source_engine": str(getattr(atom, "source_engine", "") or ""),
        "metadata":      safe_json(getattr(atom, "metadata", {})) or {},
    }


def _atom_to_edge(atom: Any) -> dict | None:
    outgoing = list(getattr(atom, "outgoing", []))
    if len(outgoing) < 2:
        return None
    tv = getattr(atom, "truth_value", None)
    return {
        "id":         getattr(atom, "atom_id", ""),
        "source":     outgoing[0],
        "target":     outgoing[1],
        "label":      str(getattr(atom, "atom_type", "")),
        "strength":   float(getattr(tv, "strength",   0.5)) if tv else 0.5,
        "confidence": float(getattr(tv, "confidence", 0.0)) if tv else 0.0,
        "weight":     float(getattr(tv, "strength",   0.5)) if tv else 0.5,
    }


@app.get("/map/atomspace")
async def get_map_atomspace():
    """All AtomSpace atoms as graph nodes/edges."""
    stack = _require_stack()
    engine = _find_atomspace_engine(stack)
    if engine is None:
        return {"nodes": [], "edges": [], "atom_count": 0,
                "node_count": 0, "edge_count": 0, "status": "atomspace_not_found"}
    try:
        atoms = engine.get_all_atoms()
    except Exception as exc:
        log.exception("AtomSpace get_all_atoms failed.")
        raise HTTPException(status_code=500, detail=str(exc))
    nodes, edges, node_ids = [], [], set()
    for atom in atoms:
        if not list(getattr(atom, "outgoing", [])):
            nodes.append(_atom_to_node(atom))
            node_ids.add(getattr(atom, "atom_id", ""))
        else:
            e = _atom_to_edge(atom)
            if e:
                edges.append(e)
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
    return {"nodes": nodes, "edges": edges, "atom_count": len(atoms),
            "node_count": len(nodes), "edge_count": len(edges), "status": "ok"}


@app.get("/map/knowledge_maps")
async def get_knowledge_maps_list():
    """List all KnowledgeMap summaries."""
    stack = _require_stack()
    store = getattr(getattr(stack.memory, "knowledge", None), "knowledge_maps", None)
    if store is None:
        return _items([])
    try:
        maps = store.get_all()
    except Exception:
        return _items([])
    return _items([{
        "map_id":           getattr(m, "map_id",           ""),
        "title":            getattr(m, "title",            ""),
        "subject_category": getattr(m, "subject_category", ""),
        "description":      (getattr(m, "description",     "") or "")[:200],
        "node_count":       len(getattr(m, "nodes", [])),
        "edge_count":       len(getattr(m, "links", [])),
        "tags":             safe_json(getattr(m, "tags", [])) or [],
        "last_updated":     str(getattr(m, "last_updated", "")),
    } for m in maps])


@app.get("/map/knowledge_maps/{map_id}/graph")
async def get_knowledge_map_graph(map_id: str):
    """Single KnowledgeMap as graph nodes/edges."""
    stack = _require_stack()
    store = getattr(getattr(stack.memory, "knowledge", None), "knowledge_maps", None)
    if store is None:
        raise HTTPException(status_code=404, detail="KnowledgeMapStore not available.")
    m = store.get_by_id(map_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Map {map_id!r} not found.")
    nodes = [{
        "id":         getattr(n, "node_id",   ""),
        "label":      getattr(n, "label",     ""),
        "type":       getattr(n, "node_type", "concept"),
        "strength":   float(getattr(n, "confidence", 0.5)),
        "confidence": float(getattr(n, "confidence", 0.5)),
        "sti": 0.0, "lti": 0.0, "source_engine": "", "metadata": {},
    } for n in getattr(m, "nodes", [])]
    edges = [{
        "id":         getattr(l, "link_id",     ""),
        "source":     getattr(l, "source_node", ""),
        "target":     getattr(l, "target_node", ""),
        "label":      getattr(l, "relation",    ""),
        "strength":   float(getattr(l, "weight", 0.5)),
        "confidence": 1.0,
        "weight":     float(getattr(l, "weight", 0.5)),
    } for l in getattr(m, "links", [])]
    return {"map_id": getattr(m, "map_id", ""), "title": getattr(m, "title", ""),
            "nodes": nodes, "edges": edges,
            "node_count": len(nodes), "edge_count": len(edges), "status": "ok"}
