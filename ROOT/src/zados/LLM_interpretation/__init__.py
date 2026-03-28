"""
ZA-DOS LLM Interpretation Layer (v0.5).

Public API
----------
LLMInterpretationLayer
    Main entry point.  Call .run(stmm, input_bundle) after reward
    evaluation completes.  Returns the user-facing response string.

Phase5Evaluator
    Two-pathway reward evaluation wrapper.  Used internally by the
    layer but exposed for pipeline-level access if needed.

Phase5Result
    Dataclass holding the output of Phase 5 evaluation.
"""
from zados.LLM_interpretation.llm_layer import LLMInterpretationLayer
from zados.LLM_interpretation.phase5_evaluator import Phase5Evaluator, Phase5Result

__all__ = [
    "LLMInterpretationLayer",
    "Phase5Evaluator",
    "Phase5Result",
]
