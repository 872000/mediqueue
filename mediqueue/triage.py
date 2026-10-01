"""Priority triage rules.

Urgent tickets jump ahead of followup/general, but a fairness cap prevents
starvation: at most MAX_CONSECUTIVE_URGENT urgent tickets may be served in a
row before a non-urgent ticket must be served.
"""

MAX_CONSECUTIVE_URGENT = 2

# Base service-time distributions (minutes) per category: (mean, std)
SERVICE_TIME = {
    "general": (18.0, 5.0),
    "followup": (10.0, 3.0),
    "urgent": (14.0, 6.0),
}

PRIORITY = {"urgent": 0, "followup": 1, "general": 2}


def pick_next(waiting, consecutive_urgent):
    """Choose the next ticket to serve from the waiting list.

    waiting: list of Ticket (or any object with .category and .arrived_at),
             in arrival order.
    consecutive_urgent: how many urgent tickets were served consecutively
                        right before this pick.

    Returns (ticket, bumped_by_cap) or (None, False) if the queue is empty.
    bumped_by_cap is True when the fairness cap forced an urgent ticket
    to wait for a non-urgent one.
    """
    if not waiting:
        return None, False
    urgents = [t for t in waiting if t.category == "urgent"]
    others = [t for t in waiting if t.category != "urgent"]
    if urgents and (not others or consecutive_urgent < MAX_CONSECUTIVE_URGENT):
        return urgents[0], False
    if others:
        # followup before general within non-urgent (higher priority first)
        others.sort(key=lambda t: (PRIORITY[t.category], t.arrived_at))
        return others[0], bool(urgents and consecutive_urgent >= MAX_CONSECUTIVE_URGENT)
    return urgents[0], False


def predict_service_time(category, rng):
    mean, std = SERVICE_TIME[category]
    return max(2.0, rng.normal(mean, std))
