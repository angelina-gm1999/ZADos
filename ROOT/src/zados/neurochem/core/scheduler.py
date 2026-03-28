from typing import Callable, List, Tuple


class EventScheduler:
    """
    Schedules timed events (e.g., modulations, inputs, switches)
    during neurochemical simulation.

    Useful for spike-triggered modulation or domain coupling.
    """

    def __init__(self):
        self._events: List[Tuple[float, Callable]] = []

    def add_event(self, time: float, action: Callable):
        """
        Add an event to run at a specific time.

        Parameters
        ----------
        time : float
            Simulation time at which to trigger the event
        action : Callable
            Function to execute at that time
        """
        self._events.append((time, action))
        self._events.sort(key=lambda x: x[0])  # Keep sorted

    def trigger_events(self, current_time: float):
        """
        Run all scheduled events at this timestep.

        Parameters
        ----------
        current_time : float
            Current simulation time
        """
        while self._events and self._events[0][0] <= current_time:
            _, action = self._events.pop(0)
            action()
