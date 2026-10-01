"""Wait-time predictor: RandomForestRegressor trained on synthetic history.

Features for a new ticket:
  [queue_general, queue_followup, queue_urgent, hour_of_day, category_code]
Target: minutes until service starts (predicted wait).
"""

import csv
import os
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .triage import SERVICE_TIME

CATEGORIES = ["general", "followup", "urgent"]
CATEGORY_CODE = {c: i for i, c in enumerate(CATEGORIES)}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_history.csv")


@dataclass
class Prediction:
    minutes: float
    model_name: str = "RandomForestRegressor(n_estimators=200)"


class WaitPredictor:
    MODEL_NAME = "RandomForestRegressor(n_estimators=200)"

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.trained = False

    @staticmethod
    def features(queue_counts, hour_of_day, category):
        return np.array([[
            queue_counts.get("general", 0),
            queue_counts.get("followup", 0),
            queue_counts.get("urgent", 0),
            hour_of_day,
            CATEGORY_CODE[category],
        ]], dtype=float)

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def train_from_csv(self, path=DATA_PATH):
        X, y = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                X.append([
                    float(row["queue_general"]), float(row["queue_followup"]),
                    float(row["queue_urgent"]), float(row["hour_of_day"]),
                    float(row["category_code"]),
                ])
                y.append(float(row["wait_minutes"]))
        self.train(np.array(X), np.array(y))
        return len(y)

    def predict(self, queue_counts, hour_of_day, category) -> Prediction:
        if not self.trained:
            raise RuntimeError("predictor not trained")
        minutes = float(self.model.predict(
            self.features(queue_counts, hour_of_day, category))[0])
        return Prediction(minutes=max(0.0, minutes))

    def predict_many(self, queue_counts, hour_of_day, categories) -> dict:
        """Batched prediction: one model call for many (category) queries."""
        if not self.trained:
            raise RuntimeError("predictor not trained")
        X = np.vstack([self.features(queue_counts, hour_of_day, c)[0]
                       for c in categories])
        minutes = self.model.predict(X)
        return {c: max(0.0, float(m)) for c, m in zip(categories, minutes)}


def generate_synthetic_data(path=DATA_PATH, n=4000, seed=7):
    """Create bundled synthetic historical data and save to CSV."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        hour = rng.uniform(8, 17)
        qg = rng.integers(0, 12)
        qf = rng.integers(0, 6)
        qu = rng.integers(0, 3)
        cat = CATEGORIES[rng.integers(0, 3)]
        # Ground truth: expected service backlog ahead + noise.
        # Urgent tickets mostly wait only for other urgent work.
        svc = {c: SERVICE_TIME[c][0] for c in CATEGORIES}
        if cat == "urgent":
            backlog = qu * svc["urgent"] * 0.5
        elif cat == "followup":
            backlog = qu * svc["urgent"] + qf * svc["followup"] * 0.8
        else:
            backlog = (qu * svc["urgent"] + qf * svc["followup"]
                       + qg * svc["general"]) * 0.9
        wait = max(1.0, backlog + rng.normal(0, 3))
        rows.append({
            "queue_general": qg, "queue_followup": qf, "queue_urgent": qu,
            "hour_of_day": round(hour, 2), "category": cat,
            "category_code": CATEGORY_CODE[cat],
            "wait_minutes": round(wait, 1),
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path, n
