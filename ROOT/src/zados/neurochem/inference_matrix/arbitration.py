"""
Bidirectional arbitration orchestrator.

The InferenceArbitrator coordinates the NT→engine→NT loop:
1. Read neurosymbolic metrics
2. Compute engine priority weights (nt_to_engine)
3. (External: cognitive engines run with these priorities)
4. Process evaluation results (engine_to_nt)
5. Return modulation signals for the next engine step
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from zados.neurochem.inference_matrix.nt_to_engine import (
    compute_engine_priority_weights,
    EnginePriorityWeights,
)
from zados.neurochem.inference_matrix.engine_to_nt import (
    compute_nt_modulation_from_evaluation,
)


class InferenceArbitrator:
    """
    Orchestrates the bidirectional NT ↔ cognitive engine loop.

    Usage
    -----
    >>> arb = InferenceArbitrator()
    >>> weights = arb.compute_priorities(metrics)
    >>> # ... cognitive engines run with these weights ...
    >>> modulation = arb.process_evaluation(eval_results, metrics)
    >>> engine.step(modulation)  # feed back into neurochemical engine
    """

    def __init__(self) -> None:
        self._last_priorities: Optional[EnginePriorityWeights] = None
        self._last_modulation: Optional[Dict[str, Dict[str, float]]] = None
        self._step_count: int = 0

    @property
    def last_priorities(self) -> Optional[EnginePriorityWeights]:
        """Most recently computed engine priority weights."""
        return self._last_priorities

    @property
    def last_modulation(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Most recently computed NT modulation signals."""
        return self._last_modulation

    @property
    def step_count(self) -> int:
        """Number of arbitration cycles completed."""
        return self._step_count

    def compute_priorities(
        self,
        metrics: Dict[str, float],
    ) -> EnginePriorityWeights:
        """
        Phase 1: Compute cognitive engine priority weights from NT metrics.

        Parameters
        ----------
        metrics : dict
            Neurosymbolic metrics (motivation, empathy, etc.)

        Returns
        -------
        EnginePriorityWeights
            Priority weights for each cognitive engine
        """
        weights = compute_engine_priority_weights(metrics)
        self._last_priorities = weights
        return weights

    def process_evaluation(
        self,
        evaluation_results: Dict[str, Any],
        current_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Phase 2: Process evaluation results into NT modulation signals.

        Parameters
        ----------
        evaluation_results : dict
            Results from cognitive engine evaluation
        current_metrics : dict, optional
            Current neurosymbolic metrics for adaptive feedback

        Returns
        -------
        dict
            {nt_name: {signal_key: value}} for engine.step()
        """
        modulation = compute_nt_modulation_from_evaluation(
            evaluation_results, current_metrics,
        )
        self._last_modulation = modulation
        self._step_count += 1
        return modulation

    def full_cycle(
        self,
        metrics: Dict[str, float],
        evaluation_results: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """
        Run a complete arbitration cycle.

        Computes priorities (for external use) and processes evaluation
        results into modulation signals.

        Parameters
        ----------
        metrics : dict
            Current neurosymbolic metrics
        evaluation_results : dict
            Results from cognitive engine evaluation

        Returns
        -------
        dict
            NT modulation signals for the next engine step
        """
        self.compute_priorities(metrics)
        return self.process_evaluation(evaluation_results, metrics)

    def reset(self) -> None:
        """Reset arbitrator state."""
        self._last_priorities = None
        self._last_modulation = None
        self._step_count = 0
