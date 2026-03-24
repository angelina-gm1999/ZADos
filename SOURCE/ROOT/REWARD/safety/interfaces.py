from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ConstraintHookInterface(ABC):
    """
    Hard constraint hook.

    Constraint hooks have absolute priority over reward modulation.
    Reward signals can never override constraint outcomes.
    """

    @abstractmethod
    def check(
        self,
        *,
        state: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate constraints against a proposed state.

        Must return:
            allowed : bool
            action  : one of {"allow", "veto", "rollback", "revert"}
            reason  : optional string
        """
        raise NotImplementedError
