"""CLI: `python3 -m mediqueue.cli demo` runs the full clinic-day simulation."""

import argparse
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "..", "data", "synthetic_history.csv")
OUT_DIR = os.path.join(ROOT, "..", "screenshots")


def cmd_demo(args):
    from .predictor import WaitPredictor, generate_synthetic_data
    from .simulate import simulate_day
    from .dashboard import (chart_predicted_vs_actual, chart_queue_timeline,
                            write_dashboard)

    if not os.path.exists(DATA_CSV) or args.regen_data:
        path, n = generate_synthetic_data(DATA_CSV)
        print(f"Generated bundled synthetic dataset: {n} rows -> {path}")

    predictor = WaitPredictor()
    n = predictor.train_from_csv(DATA_CSV)
    print(f"Trained {predictor.MODEL_NAME} on {n} synthetic history rows.\n")
    print("=" * 70)
    print(" MediQueue — simulated clinic day".center(70))
    print("=" * 70)

    sim, stats = simulate_day(predictor, seed=args.seed, verbose=args.verbose)

    print("\n" + "=" * 70)
    print(" Predicted vs actual wait comparison")
    print("=" * 70)
    print(f" Patients served      : {stats['served']}")
    print(f" Avg predicted wait   : {stats['avg_predicted']:.1f} min")
    print(f" Avg actual wait      : {stats['avg_actual']:.1f} min")
    print(f" Mean abs. error      : {stats['mae_minutes']:.1f} min")
    print(f" Max consec. urgent   : {stats['max_consecutive_urgent']} (fairness cap = 2)")

    os.makedirs(OUT_DIR, exist_ok=True)
    p1 = chart_predicted_vs_actual(sim.predictions,
                                   os.path.join(OUT_DIR, "predicted_vs_actual.png"))
    p2 = chart_queue_timeline(sim.events,
                              os.path.join(OUT_DIR, "queue_timeline.png"))
    html = write_dashboard(stats, os.path.join(OUT_DIR, "dashboard.html"), p1, p2)
    print(f"\n Charts : {p1}, {p2}")
    print(f" Dashboard: {html}")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mediqueue",
                                 description="MediQueue: smart clinic queue demo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="simulate a full clinic day")
    d.add_argument("--seed", type=int, default=123)
    d.add_argument("--verbose", action="store_true", default=True)
    d.add_argument("--quiet", dest="verbose", action="store_false")
    d.add_argument("--regen-data", action="store_true",
                   help="regenerate bundled synthetic dataset")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
