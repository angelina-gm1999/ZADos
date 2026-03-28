from zados.neurochem.core.scheduler import EventScheduler

def test_scheduler_runs_scheduled_events():
    scheduler = EventScheduler()
    triggered = []

    def event1():
        triggered.append("event1")

    def event2():
        triggered.append("event2")

    scheduler.add_event(1.0, event1)
    scheduler.add_event(2.0, event2)

    # Nothing should run yet
    scheduler.trigger_events(0.5)
    assert triggered == []

    # First event should run
    scheduler.trigger_events(1.0)
    assert triggered == ["event1"]

    # Second event should run
    scheduler.trigger_events(2.0)
    assert triggered == ["event1", "event2"]

def test_scheduler_triggers_in_order():
    scheduler = EventScheduler()
    order = []

    scheduler.add_event(3.0, lambda: order.append("late"))
    scheduler.add_event(1.0, lambda: order.append("early"))
    scheduler.add_event(2.0, lambda: order.append("middle"))

    scheduler.trigger_events(3.0)
    assert order == ["early", "middle", "late"]
