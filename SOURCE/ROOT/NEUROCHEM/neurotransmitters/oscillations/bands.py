from __future__ import annotations
from typing import Dict


class OscillationState:
    """
    Maintains current power levels across EEG bands.
    """
    def __init__(self, initial: Dict[str, float] | None = None):
        """
        Args:
            initial (dict): Optional dict of initial values for each band.
        """
        self.bands = {
            'delta': 0.0,
            'theta': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0,
        }

        if initial:
            for k, v in initial.items():
                if k in self.bands:
                    self.bands[k] = float(v)

    def set(self, band: str, value: float):
        if band in self.bands:
            self.bands[band] = float(value)

    def get(self, band: str) -> float:
        return self.bands.get(band, 0.0)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.bands)

    def normalize(self):
        total = sum(self.bands.values())
        if total > 0.0:
            for k in self.bands:
                self.bands[k] /= total
