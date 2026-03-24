from zados.reward.safety.interfaces import ConstraintHookInterface
from zados.reward.safety.reward_bridge import RewardSafetyBridge


class AllowHook(ConstraintHookInterface):
    def check(self, *, state, context):
        return {
            "allowed": True,
            "action": "allow",
            "reason": None,
        }


class VetoHook(ConstraintHookInterface):
    def check(self, *, state, context):
        return {
            "allowed": False,
            "action": "veto",
            "reason": "constraint violation",
        }


def test_allows_state_when_constraints_pass():
    bridge = RewardSafetyBridge(hooks=[AllowHook()])

    result = bridge.evaluate(
        proposed_state={"x": 1},
        reward_signal={"reward": 0.8},
        context={},
    )

    assert result["allowed"] is True
    assert result["final_state"] == {"x": 1}
    assert result["action"] == "allow"


def test_veto_reverts_to_last_verified_state():
    bridge = RewardSafetyBridge(hooks=[VetoHook()])
    bridge.register_verified_state({"x": 0})

    result = bridge.evaluate(
        proposed_state={"x": 999},
        reward_signal={"reward": 100.0},
        context={},
    )

    assert result["allowed"] is False
    assert result["final_state"] == {"x": 0}
    assert result["action"] == "veto"
    assert result["reason"] == "constraint violation"
