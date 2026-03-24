from __future__ import annotations

from typing import Any, Dict, List, Optional

from zados.reward.safety.interfaces import ConstraintHookInterface


class RewardSafetyBridge:
    """
    Safety enforcement layer between reward synthesis and state acceptance.

    Guarantees:
    - Constraints dominate reward
    - Unsafe states cannot be committed
    - Last verified state can be restored
    """

    def __init__(self, hooks: List[ConstraintHookInterface]):
        self._hooks = list(hooks)
        self._last_verified_state: Optional[Any] = None

    def register_verified_state(self, state: Any) -> None:
        """
        Store last known-good state snapshot.
        """
        self._last_verified_state = state

    def evaluate(
        self,
        *,
        proposed_state: Any,
        reward_signal: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate constraint hooks before accepting reward-modulated state.

        Returns:
            allowed     : bool
            final_state : Any
            action      : str
            reason      : optional str
        """

        for hook in self._hooks:
            result = hook.check(
                state=proposed_state,
                context=context,
            )

            allowed = bool(result.get("allowed", False))
            action = result.get("action", "veto")
            reason = result.get("reason")

            if not allowed:
                if action not in {"veto", "rollback", "revert"}:
                    raise ValueError(f"Unknown constraint action: {action}")

                return {
                    "allowed": False,
                    "final_state": self._last_verified_state,
                    "action": action,
                    "reason": reason,
                }

        # All constraints passed
        self.register_verified_state(proposed_state)

        return {
            "allowed": True,
            "final_state": proposed_state,
            "action": "allow",
            "reason": None,
        }
