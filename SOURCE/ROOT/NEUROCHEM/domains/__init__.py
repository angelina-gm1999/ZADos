"""
Domain-to-neurotransmitter mapping modules.

Each domain module defines how cognitive evaluation domain subscores
map to neurotransmitter modulation signals. These formalize the
implicit mappings in the reward adapter's mapping.py.

Domains:
    Innovation → DA, CB1 (novelty, flexibility)
    Logic → NE, ACh, GLU (precision, attention, integration)
    Human Attunement → OXT, 5HT, MOR (empathy, mood, comfort)
    Ethics → GABA, cortisol, CRH (inhibition, stress, boundaries)
"""

from .base import DomainNTMapping, NTSignalMapping

__all__ = [
    "DomainNTMapping",
    "NTSignalMapping",
]
