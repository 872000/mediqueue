"""Real tests for MediQueue: triage ordering, fairness cap, predictor, full-day sim."""

from collections import namedtuple

import numpy as np
import pytest

from mediqueue.db import QueueDB
from mediqueue.predictor import WaitPredictor, generate_synthetic_data
from mediqueue.simulate import simulate_day
from mediqueue.triage import (MAX_CONSECUTIVE_URGENT, SERVICE_TIME, pick_next)

FakeTicket = namedtuple("FakeTicket", ["ticket_no", "category", "arrived_at", "id"])


def make_waiting(cats):
    return [FakeTicket(f"T{i:03d}", c, float(i), i) for i, c in enumerate(cats)]


# --- triage ordering --------------------------------------------------------
def test_urgent_jumps_ahead_of_general_and_followup():
    waiting = make_waiting(["general", "followup", "urgent", "general"])
    ticket, _ = pick_next(waiting, consecutive_urgent=0)
    assert ticket.category == "urgent"


def test_followup_before_general_when_no_urgent():
    waiting = make_waiting(["general", "followup", "general"])
    ticket, _ = pick_next(waiting, consecutive_urgent=0)
    assert ticket.category == "followup"


def test_fifo_within_same_category():
    waiting = make_waiting(["general", "general", "general"])
    ticket, _ = pick_next(waiting, consecutive_urgent=0)
    assert ticket.ticket_no == "T000"


def test_empty_queue_returns_none():
    ticket, capped = pick_next([], 0)
    assert ticket is None and capped is False


# --- fairness cap -----------------------------------------------------------
def test_fairness_cap_blocks_third_consecutive_urgent():
    waiting = make_waiting(["general", "urgent", "urgent"])
    ticket, capped = pick_next(waiting, consecutive_urgent=MAX_CONSECUTIVE_URGENT)
    assert ticket.category == "general"
    assert capped is True


def test_two_consecutive_urgent_allowed():
    waiting = make_waiting(["general", "urgent"])
    t1, _ = pick_next(waiting, consecutive_urgent=0)
    assert t1.category == "urgent"
    t2, _ = pick_next(waiting, consecutive_urgent=1)
    assert t2.category == "urgent"


def test_cap_resets_after_non_urgent_served():
    # after a non-urgent is served, consecutive_urgent drops to 0
    waiting = make_waiting(["urgent"])
    ticket, _ = pick_next(waiting, consecutive_urgent=0)
    assert ticket.category == "urgent"


def test_all_urgent_queue_still_serves_when_capped():
    # cap must not deadlock when only urgent tickets remain
    waiting = make_waiting(["urgent", "urgent"])
    ticket, capped = pick_next(waiting, consecutive_urgent=99)
    assert ticket is not None and ticket.category == "urgent"


# --- predictor --------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "history.csv"
    generate_synthetic_data(str(path), n=2000, seed=99)
    p = WaitPredictor()
    assert p.train_from_csv(str(path)) == 2000
    return p


def test_predictor_trains_and_predicts(trained_predictor):
    pred = trained_predictor.predict(
        {"general": 3, "followup": 1, "urgent": 0}, 10.5, "general")
    assert 0 <= pred.minutes <= 200


def test_predictor_monotonic_in_queue_length(trained_predictor):
    empty = trained_predictor.predict(
        {"general": 0, "followup": 0, "urgent": 0}, 10.5, "general").minutes
    busy = trained_predictor.predict(
        {"general": 10, "followup": 5, "urgent": 2}, 10.5, "general").minutes
    assert busy > empty


def test_predictor_urgent_waits_less_than_general(trained_predictor):
    counts = {"general": 5, "followup": 2, "urgent": 1}
    u = trained_predictor.predict(counts, 10.5, "urgent").minutes
    g = trained_predictor.predict(counts, 10.5, "general").minutes
    assert u < g


def test_predictor_requires_training():
    with pytest.raises(RuntimeError):
        WaitPredictor().predict({"general": 1}, 9.0, "general")


# --- database ---------------------------------------------------------------
def test_db_ticket_lifecycle():
    db = QueueDB()
    tid = db.add_ticket("T001", "general", arrived_at=480.0, predicted_wait=12.0)
    assert len(db.waiting()) == 1
    assert db.waiting_counts()["general"] == 1
    db.set_status(tid, "serving", service_start=490.0)
    assert len(db.waiting()) == 0
    db.set_status(tid, "served", service_end=508.0)
    assert db.served_count() == 1
    t = db.all_tickets()[0]
    assert t.ticket_no == "T001" and t.service_end == 508.0
    db.close()


# --- full-day simulation ----------------------------------------------------
@pytest.fixture(scope="module")
def day_result(trained_predictor):
    sim, stats = simulate_day(trained_predictor, seed=123, verbose=False)
    return sim, stats


def test_simulation_completes_and_serves_patients(day_result):
    _, stats = day_result
    assert stats["served"] > 20  # a real clinic day, not a stub


def test_simulation_stats_consistent(day_result):
    sim, stats = day_result
    assert stats["served"] == len(sim.predictions)  # every served ticket tracked
    assert 0 <= stats["avg_predicted"] <= 240
    assert 0 <= stats["avg_actual"] <= 240
    assert stats["mae_minutes"] >= 0


def test_simulation_predictions_are_sane(day_result):
    sim, _ = day_result
    assert len(sim.predictions) > 0
    for ticket_no, category, predicted, actual in sim.predictions:
        assert category in SERVICE_TIME
        assert predicted >= 0
        assert actual >= 0


def test_simulation_respects_fairness_cap(day_result):
    _, stats = day_result
    assert stats["max_consecutive_urgent"] <= MAX_CONSECUTIVE_URGENT


def test_simulation_mae_is_reasonable(day_result):
    _, stats = day_result
    assert stats["mae_minutes"] < 25  # model should beat a dumb baseline
