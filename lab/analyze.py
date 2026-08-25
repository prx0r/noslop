"""Analyze ablation results: per-arm reduction stats with Wilson CIs + paired wins.

Reads experiments/lab/results.jsonl, groups by stage|payload(|mode),
reports mean reduction, retention, clean-rate with Wilson intervals,
and paired comparisons vs the 'none' control arm.

Run: python3 lab/analyze.py [--stage A]
"""
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "experiments", "lab", "results.jsonl")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def load(stage=None):
    rows = []
    if not os.path.exists(RESULTS):
        return rows
    for line in open(RESULTS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if stage is None or r["config"]["stage"] == stage:
            rows.append(r)
    return rows


def main():
    stage = None
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    rows = load(stage)
    if not rows:
        print("no results yet")
        return

    arms = defaultdict(list)
    for r in rows:
        cfg = r["config"]
        arm = f"{cfg['payload']}|{cfg['mode']}" if cfg["stage"] == "B" else cfg["payload"]
        arms[arm].append(r)

    print(f"{'arm':<22} {'n':>2} {'red%':>7} {'ret%':>6} {'clean':>11} "
          f"{'calls/ep':>8} {'wall_s':>7}")
    order = ["none", "counts", "locate", "highlight", "full"]
    keys = ([k for k in order if k in arms] +
            [k for k in arms if k not in order])
    summary = {}
    for arm in keys:
        rs = arms[arm]
        n = len(rs)
        red = sum(r["pattern_reduction_pct"] for r in rs) / n
        ret = sum(r["length_retention_pct"] for r in rs) / n
        clean = sum(1 for r in rs if r["clean"])
        p, lo, hi = wilson(clean, n)
        calls = sum(r["hermes_calls_total"] - 1 for r in rs) / n  # excl draft
        wall = sum(r["wall_s"] for r in rs) / n
        print(f"{arm:<22} {n:>2} {red:>6.1f}% {ret:>5.1f}% "
              f"{clean:>2}/{n} [{lo:.0%}-{hi:.0%}] {calls:>8.1f} {wall:>7.0f}")
        summary[arm] = {"n": n, "mean_reduction": round(red, 1),
                        "mean_retention": round(ret, 1), "clean": clean,
                        "clean_wilson": [round(lo, 3), round(hi, 3)],
                        "mean_extra_calls": round(calls, 2)}

    # Paired comparison vs none (same topics shared across arms)
    if "none" in arms:
        def topic_of(r):
            return r["key"].split("|")[-1]
        none_by_topic = {topic_of(r): r for r in arms["none"]}
        print("\npaired vs 'none' control (same drafts):")
        for arm in keys:
            if arm == "none":
                continue
            wins = ties = losses = 0
            for r in arms[arm]:
                ctrl = none_by_topic.get(topic_of(r))
                if not ctrl:
                    continue
                d = r["pattern_reduction_pct"] - ctrl["pattern_reduction_pct"]
                if d > 5:
                    wins += 1
                elif d < -5:
                    losses += 1
                else:
                    ties += 1
            print(f"  {arm:<20} win/tie/loss vs none = {wins}/{ties}/{losses}")

    out = os.path.join(os.path.dirname(RESULTS), f"analysis_{stage or 'all'}.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
