"""
Utility modules for the neurochemical layer.

Provides helper functions for converting between emotion profiles
and neurochemical modulation signals.
"""

from .emotion_interface import emotion_profile_to_signals

__all__ = [
    "emotion_profile_to_signals",
]
