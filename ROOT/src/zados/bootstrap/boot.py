"""
ZADOS Stack Bootstrap — `boot_zados()`
======================================

Single-call factory that constructs the full ZADOS stack:
  - NeurochemicalEngine (with NTs, receptor modules, oscillation state)
  - MemoryLayer (STMM + MTMM + LTMM)
  - All 32 cognitive engines (E1-E32)
  - SessionOrchestrator
  - InputClassifier

Reused by:
  - dev_ui (Python REPL)
  - bridge.server (FastAPI bridge — eventually)
  - notebooks, tests, scripts

Engine construction is defensive: a single engine failure does not abort the
stack — the offending engine is logged and skipped, and Phase 3 will treat
it as not-registered.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zados.cognitive_engines.constants import ENGINE_IDS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine class name overrides
# ---------------------------------------------------------------------------
# Default convention: snake_case module name -> PascalCase class name.
# These three deviate because of uppercase abbreviations.
_CLASS_OVERRIDES: Dict[int, str] = {
    3:  "SOARProductionEngine",
    9:  "AtomSpaceEngine",
    10: "PLNEngine",
    16: "ECANEngine",
}

# Modules that live under cognitools/ instead of py_engines/
_COGNITOOLS_IDS = {9, 10, 16}


@dataclass
class ZadosStack:
    """Full ZADOS stack — all components needed for end-to-end operation."""
    orchestrator: Any            # SessionOrchestrator
    classifier: Any              # InputClassifier
    memory: Any                  # MemoryLayer
    neurochem: Any               # NeurochemicalEngine
    engines: Dict[int, Any]      # 32 cognitive engines
    engine_errors: Dict[int, str] = field(default_factory=dict)


def boot_zados(
    register_engines: bool = True,
    open_session: bool = True,
) -> ZadosStack:
    """Build the full ZADOS stack.

    Parameters
    ----------
    register_engines : bool
        If True (default), construct and register all 32 cognitive engines.
        If False, the stack is built with an empty engine dict (Phase 3 will
        skip all engine dispatches — useful for minimal-overhead smoke tests).
    open_session : bool
        If True (default), call ``orchestrator.open_session()`` before return.

    Returns
    -------
    ZadosStack
    """
    log.info("boot_zados: starting...")

    neurochem = _build_neurochem()
    memory = _build_memory()

    engines: Dict[int, Any] = {}
    engine_errors: Dict[int, str] = {}
    if register_engines:
        engines, engine_errors = _build_engines()
        log.info(
            "boot_zados: %d/%d engines registered (%d errors)",
            len(engines), len(ENGINE_IDS), len(engine_errors),
        )

    orchestrator = _build_orchestrator(neurochem, memory, engines)
    classifier = _build_classifier(orchestrator)

    if open_session:
        orchestrator.open_session()
        log.info("boot_zados: session opened (id=%s)", orchestrator.session.session_id)

    log.info("boot_zados: ready.")
    return ZadosStack(
        orchestrator=orchestrator,
        classifier=classifier,
        memory=memory,
        neurochem=neurochem,
        engines=engines,
        engine_errors=engine_errors,
    )


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def _build_neurochem() -> Any:
    from zados.neurochem import NeurochemicalEngine
    from zados.neurochem.neurotransmitters.configs import (
        register_all_neurotransmitters,
        register_all_receptor_modules_on_engine,
    )
    from zados.neurochem.neurotransmitters.module_registry import (
        register_all_nt_modules,
    )
    from zados.neurochem.receptors.receptor_registry import (
        register_all_receptor_modules,
    )

    nc = NeurochemicalEngine()
    register_all_neurotransmitters(nc, include_receptors=True)
    register_all_nt_modules(engine=nc)
    register_all_receptor_modules()
    register_all_receptor_modules_on_engine(nc)

    if nc.registry.get_oscillations() is None:
        from zados.neurochem.state.oscillation_state import OscillationState
        nc.set_oscillation_state(OscillationState())

    return nc


def _build_memory() -> Any:
    from zados.memory import MemoryLayer
    return MemoryLayer()


def _build_engines() -> Tuple[Dict[int, Any], Dict[int, str]]:
    """Construct all 32 engines. Returns (engines_dict, errors_dict)."""
    engines: Dict[int, Any] = {}
    errors: Dict[int, str] = {}

    # First pass: zero-arg engines (everything except E10, E16).
    for eid, mod_name in ENGINE_IDS.items():
        if eid in (10, 16):
            continue
        cls = _resolve_engine_class(eid, mod_name)
        if cls is None:
            errors[eid] = f"class not found in module {mod_name}"
            continue
        try:
            engines[eid] = cls()
        except Exception as exc:  # noqa: BLE001 — defensive boot
            errors[eid] = f"{type(exc).__name__}: {exc}"
            log.warning("boot_zados: engine E%d construction failed: %s", eid, exc)

    # Second pass: engines that depend on E9 (AtomSpace).
    atomspace = engines.get(9)
    for eid in (10, 16):
        cls = _resolve_engine_class(eid, ENGINE_IDS[eid])
        if cls is None:
            errors[eid] = f"class not found in module {ENGINE_IDS[eid]}"
            continue
        if atomspace is None:
            errors[eid] = "AtomSpace (E9) unavailable"
            continue
        try:
            engines[eid] = cls(atomspace)
        except Exception as exc:  # noqa: BLE001
            errors[eid] = f"{type(exc).__name__}: {exc}"
            log.warning("boot_zados: engine E%d construction failed: %s", eid, exc)

    return engines, errors


def _resolve_engine_class(eid: int, module_name: str) -> Optional[type]:
    pkg = "zados.cognitive_engines.cognitools" if eid in _COGNITOOLS_IDS \
        else "zados.cognitive_engines.py_engines"
    try:
        mod = importlib.import_module(f"{pkg}.{module_name}")
    except ImportError as exc:
        log.debug("boot_zados: cannot import %s.%s: %s", pkg, module_name, exc)
        return None
    cls_name = _CLASS_OVERRIDES.get(eid) or _to_pascal(module_name)
    return getattr(mod, cls_name, None)


def _to_pascal(snake: str) -> str:
    return "".join(p.capitalize() for p in snake.split("_"))


def _build_orchestrator(neurochem: Any, memory: Any, engines: Dict[int, Any]) -> Any:
    from zados.core.session import SessionOrchestrator

    # Resolve optional dependencies from memory if present.
    hardcoded = None
    try:
        identity = getattr(memory, "identity", None)
        hardcoded = getattr(identity, "hardcoded", None) if identity else None
    except Exception:
        pass

    return SessionOrchestrator(
        neurochem_engine=neurochem,
        memory=memory,
        engines=engines,
        hardcoded_store=hardcoded,
    )


def _build_classifier(orchestrator: Any) -> Any:
    from zados.core.main import InputClassifier
    return InputClassifier(session_orchestrator=orchestrator)


__all__ = ["boot_zados", "ZadosStack"]
