"""
ZA-DOS Core Pipeline — Phase 0: Input Bundle Reception (spec §3.2).

Validates the InputBundle before any processing begins.
"""
from __future__ import annotations

from zados.core.types import InputBundle, PipelineValidationError

# Safety tiers that block all processing.
BLOCKED_TIERS = frozenset({"FROZEN", "DREAMBOX_BANNED"})


def validate_bundle(bundle: InputBundle) -> InputBundle:
    """Validate the incoming InputBundle.

    Checks
    ------
    1. ``raw_text`` must be a non-empty string.
    2. ``safety_tier`` must not be in BLOCKED_TIERS.
    3. ``context_flags`` must not contain ``LOCKED_PIPELINE=True``.

    Returns the (unmodified) bundle on success.

    Raises
    ------
    PipelineValidationError
        If any check fails.
    """
    # 1. Required field — raw_text
    if not isinstance(bundle.raw_text, str) or not bundle.raw_text.strip():
        raise PipelineValidationError("InputBundle.raw_text must be a non-empty string.")

    # 2. Blocked safety tiers
    tier = (bundle.safety_tier or "NORMAL").upper()
    if tier in BLOCKED_TIERS:
        raise PipelineValidationError(
            f"Safety tier '{tier}' is blocked.  Processing is not permitted."
        )

    # 3. LOCKED_PIPELINE flag
    if bundle.context_flags.get("LOCKED_PIPELINE", False):
        raise PipelineValidationError(
            "Pipeline is locked (context_flags.LOCKED_PIPELINE=True)."
        )

    return bundle
