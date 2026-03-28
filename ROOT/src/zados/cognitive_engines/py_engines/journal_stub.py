"""
JournalEventStub — lightweight event-logging hook for cognitive engines.

Cognitive engines can call ``JournalEventStub.emit()`` when they detect
something notable (innovation, pattern, identity-relevant insight).  By
default this is a no-op (the stub is inactive).  The orchestration layer
wires a callback at startup to route events to the appropriate journal.

Design
------
- Event-logging only — not deep processing.  Engines call emit() and move on.
- Decoupled: engines import from constants.py or this module; they never
  depend on JournalStore, JournalWriter, or any LTMM class directly.
- The callback signature is fixed: (engine_id: str, event_type: str, data: dict).
  Any additional context the caller wants attached goes inside ``data``.

Usage (in a cognitive engine)
------------------------------
    from zados.cognitive_engines.journal_stub import journal_event_stub

    journal_event_stub.emit(
        engine_id="E19",
        event_type="innovation_flag",
        data={"pattern": "recursive_analogy", "confidence": 0.82},
    )

Wiring the callback (orchestration layer / session setup)
----------------------------------------------------------
    from zados.cognitive_engines.journal_stub import journal_event_stub

    def _on_journal_event(engine_id, event_type, data):
        ...  # route to JournalStore, log, etc.

    journal_event_stub.register(_on_journal_event)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class JournalEventStub:
    """Singleton-style event hook for cognitive engine journal events.

    Thread safety: callback registration is not thread-safe by design —
    register once at session startup before any engines are active.
    """

    def __init__(self) -> None:
        self._callbacks: List[Callable[[str, str, Dict[str, Any]], None]] = []
        self._active: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        callback: Callable[[str, str, Dict[str, Any]], None],
    ) -> None:
        """Register a callback to receive journal events.

        Parameters
        ----------
        callback : callable(engine_id, event_type, data) -> None
        """
        self._callbacks.append(callback)
        self._active = True

    def unregister_all(self) -> None:
        """Remove all callbacks (e.g. at session teardown)."""
        self._callbacks.clear()
        self._active = False

    # ------------------------------------------------------------------
    # Event emission (called from within cognitive engines)
    # ------------------------------------------------------------------

    def emit(
        self,
        engine_id: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fire a journal event.

        If no callbacks are registered this is a pure no-op.

        Parameters
        ----------
        engine_id : str
            Short engine identifier, e.g. "E7", "E14", "E19".
        event_type : str
            Semantic event type, e.g. "innovation_flag", "pattern_detected",
            "socratic_question", "identity_observation".
        data : dict, optional
            Arbitrary payload (pattern details, scores, text snippets, …).
        """
        if not self._active:
            return
        payload = data or {}
        for cb in self._callbacks:
            try:
                cb(engine_id, event_type, payload)
            except Exception:
                log.debug(
                    "JournalEventStub callback raised for engine=%s event=%s",
                    engine_id, event_type,
                )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if at least one callback is registered."""
        return self._active


# Module-level singleton — import and use directly
journal_event_stub = JournalEventStub()
