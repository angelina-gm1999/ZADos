"""
Inference matrix: bidirectional NT ↔ cognitive engine arbitration.

Modules:
    nt_to_engine: NT metrics → cognitive engine priority weights
    engine_to_nt: Evaluation results → NT modulation signals
    arbitration: Bidirectional arbitration orchestrator
"""

from .nt_to_engine import compute_engine_priority_weights
from .engine_to_nt import compute_nt_modulation_from_evaluation
from .arbitration import InferenceArbitrator

__all__ = [
    "compute_engine_priority_weights",
    "compute_nt_modulation_from_evaluation",
    "InferenceArbitrator",
]
