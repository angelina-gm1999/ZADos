"""
ZADOS Bridge Server — Stack Construction.

Builds and returns the full ZADOS processing stack.
Call build_stack() once at server startup; all components are stateful
and shared across requests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

log = logging.getLogger(__name__)


@dataclass
class ZADOSStack:
    orchestrator: Any   # SessionOrchestrator
    classifier: Any     # InputClassifier
    memory: Any         # MemoryLayer
    neurochem: Any      # NeurochemicalEngine


def build_stack() -> ZADOSStack:
    """Construct the full ZADOS stack and return it.

    Import order matters: neurochem first, then memory, then core.
    All imports are deferred so import errors surface with clear tracebacks.
    """
    log.info("Building ZADOS stack…")

    # --- Neurochemical engine ---
    from zados.neurochem import NeurochemicalEngine
    from zados.neurochem.neurotransmitters.configs import (
        register_all_neurotransmitters,
        register_all_receptor_modules_on_engine,
    )
    from zados.neurochem.neurotransmitters.module_registry import register_all_nt_modules
    from zados.neurochem.receptors.receptor_registry import register_all_receptor_modules

    neurochem = NeurochemicalEngine()
    register_all_neurotransmitters(neurochem, include_receptors=True)
    register_all_nt_modules(engine=neurochem)
    register_all_receptor_modules()          # populates static registry
    register_all_receptor_modules_on_engine(neurochem)
    log.info("NeurochemicalEngine ready.")

    # --- Memory layer ---
    from zados.memory import MemoryLayer
    memory = MemoryLayer()
    log.info("MemoryLayer ready.")

    # --- Engines dict ---
    # Phase 3 gracefully skips engines not registered here, so starting
    # with an empty dict is safe.  Add engine instances as needed:
    #   engines[23] = IntentionMapEngine()
    engines: Dict[int, Any] = {}

    # --- Session orchestrator ---
    from zados.core.session import SessionOrchestrator
    orchestrator = SessionOrchestrator(
        neurochem_engine=neurochem,
        memory=memory,
        engines=engines,
    )

    # --- Input classifier (Matrioshka outer layer) ---
    from zados.core.main import InputClassifier
    classifier = InputClassifier(session_orchestrator=orchestrator)

    log.info("ZADOS stack ready.")
    return ZADOSStack(
        orchestrator=orchestrator,
        classifier=classifier,
        memory=memory,
        neurochem=neurochem,
    )
