"""Full clinic-day simulation: arrivals, triage service, SMS-style alerts."""

import numpy as np

from .db import QueueDB
from .triage import pick_next, predict_service_time

OPEN_MIN = 8 * 60      # 08:00
CLOSE_MIN = 17 * 60    # 17:00
SMS_LEAD_MIN = 10.0    # "your turn in ~10 min" alert threshold


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def sms(ticket_no, category, minutes_left, predicted):
    print(f"[SMS] -> patient {ticket_no} ({category}): your turn in ~{minutes_left:.0f} min "
          f"(predicted wait was {predicted:.0f} min)")


class ClinicDay:
    """Simulates one clinic day minute-by-minute."""

    def __init__(self, predictor, db=None, seed=123, arrival_rate=0.28,
                 urgent_share=0.15, followup_share=0.25, verbose=True,
                 walkin_prob=0.008):
        self.predictor = predictor
        self.db = db or QueueDB()
        self.rng = np.random.default_rng(seed)
        self.arrival_rate = arrival_rate
        self.urgent_share = urgent_share
        self.followup_share = followup_share
        self.walkin_prob = walkin_prob
        self.verbose = verbose
        self.ticket_seq = 0
        self.events = []            # (time, message) timeline
        self.predictions = []       # (ticket_no, category, predicted, actual)
        self.consec_urgent = 0
        self.serving = None         # (ticket, service_end)
        self.max_consecutive_urgent_seen = 0

    # -- internals ---------------------------------------------------------
    def _log(self, t, msg):
        self.events.append((t, msg))
        if self.verbose:
            print(f"{fmt(t)}  {msg}")

    def _next_category(self):
        r = self.rng.random()
        if r < self.urgent_share:
            return "urgent"
        if r < self.urgent_share + self.followup_share:
            return "followup"
        return "general"

    def _admit(self, t, category, walkin=False):
        self.ticket_seq += 1
        ticket_no = f"{'U' if walkin else ''}T{self.ticket_seq:03d}"
        counts = self.db.waiting_counts()
        predicted = self.predictor.predict(counts, t / 60.0, category).minutes
        tid = self.db.add_ticket(ticket_no, category, t, predicted)
        kind = "urgent WALK-IN" if walkin else f"arrival ({category})"
        self._log(t, f"Ticket {ticket_no} taken: {kind} | predicted wait {predicted:.0f} min")
        return tid, ticket_no, predicted

    def _start_service(self, t, ticket):
        self.db.set_status(ticket.id, "serving", service_start=t)
        service_time = predict_service_time(ticket.category, self.rng)
        self.serving = (ticket, t, t + service_time)
        if ticket.category == "urgent":
            self.consec_urgent += 1
            self.max_consecutive_urgent_seen = max(
                self.max_consecutive_urgent_seen, self.consec_urgent)
        else:
            self.consec_urgent = 0
        self._log(t, f"Doctor starts {ticket.ticket_no} ({ticket.category}), "
                     f"~{service_time:.0f} min")

    # -- main loop ---------------------------------------------------------
    def run(self):
        t = OPEN_MIN
        # seed a small morning backlog
        for _ in range(4):
            self._admit(t, self._next_category())
        self._log(t, "Clinic opens. Doctor starts triage.")

        while True:
            # urgent walk-ins sprinkled through the day
            if t < CLOSE_MIN and self.rng.random() < self.walkin_prob:
                self._admit(t, "urgent", walkin=True)
            # regular arrivals during opening hours
            if t < CLOSE_MIN and self.rng.random() < self.arrival_rate / 5:
                self._admit(t, self._next_category())

            # doctor picks next ticket if free
            if self.serving is None:
                ticket, capped = pick_next(self.db.waiting(), self.consec_urgent)
                if ticket is not None:
                    if capped:
                        self._log(t, f"Fairness cap: {ticket.ticket_no} ({ticket.category}) "
                                     "served before waiting urgent cases")
                    self._start_service(t, ticket)

            # finish current service
            if self.serving is not None:
                ticket, start, end = self.serving
                if t >= end:
                    self.db.set_status(ticket.id, "served", service_end=t)
                    actual_wait = start - ticket.arrived_at
                    self.predictions.append(
                        (ticket.ticket_no, ticket.category,
                         ticket.predicted_wait or 0.0, actual_wait))
                    self._log(t, f"{ticket.ticket_no} done | predicted "
                                 f"{(ticket.predicted_wait or 0):.0f} min, actual {actual_wait:.0f} min")
                    self.serving = None

            # SMS alerts: warn waiting patients ~10 min before their turn.
            # Batched: one model call per category per minute, not per ticket.
            if self.serving is not None:
                waiting_now = self.db.waiting()
                if waiting_now:
                    counts = self.db.waiting_counts()
                    est_by_cat = self.predictor.predict_many(
                        counts, t / 60.0, ["general", "followup", "urgent"])
                    for w in waiting_now:
                        if w.notified:
                            continue
                        est = est_by_cat[w.category]
                        if est <= SMS_LEAD_MIN:
                            sms(w.ticket_no, w.category, est, w.predicted_wait or 0)
                            self.db.mark_notified(w.id)
                            self._log(t, f"SMS alert sent to {w.ticket_no}")

            # end condition: clinic closed and nobody waiting/being served
            if t >= CLOSE_MIN and not self.db.waiting() and self.serving is None:
                break
            t += 1
            if t > CLOSE_MIN + 300:  # hard safety stop
                break

        self._log(t, f"Clinic closes. Served {self.db.served_count()} patients.")
        return self.stats()

    def stats(self):
        preds = np.array([p for _, _, p, _ in self.predictions])
        actuals = np.array([a for _, _, _, a in self.predictions])
        err = np.abs(preds - actuals)
        return {
            "served": len(self.predictions),
            "avg_predicted": float(preds.mean()) if len(preds) else 0.0,
            "avg_actual": float(actuals.mean()) if len(actuals) else 0.0,
            "mae_minutes": float(err.mean()) if len(err) else 0.0,
            "max_consecutive_urgent": self.max_consecutive_urgent_seen,
        }


def simulate_day(predictor, seed=123, verbose=True, arrival_rate=0.28):
    sim = ClinicDay(predictor, seed=seed, verbose=verbose, arrival_rate=arrival_rate)
    stats = sim.run()
    return sim, stats
