"""Large-benchmark runner for the NoSlop miner.

Benchmarks:
  - hc3_bench.json        1370 real human/ChatGPT answers (Hello-SimpleAI/HC3)
  - polished_ai_bench.json  39 anti-slop-edited AI essays (adversarial: expect misses)
  - ../test_dataset.json    55-entry original slop-style set (in-distribution)

Reports accuracy with Wilson95, FPR/FNR per source, latency percentiles.
Run: python3 bench_run.py
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "miner"))
from miner.src.classifier import classify

HERE = os.path.dirname(__file__)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))
    return sorted_vals[idx]


def run(name, entries):
    lat, by_source = [], {}
    for e in entries:
        t0 = time.perf_counter()
        cls = classify(e["text"])
        dt = (time.perf_counter() - t0) * 1000
        pred = cls["answer"]
        lat.append(dt)
        s = by_source.setdefault(e["source"], {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
        s["n"] += 1
        if e["label"] == 1 and pred == 1:
            s["tp"] += 1
        elif e["label"] == 0 and pred == 0:
            s["tn"] += 1
        elif e["label"] == 0:
            s["fp"] += 1
        else:
            s["fn"] += 1

    n = len(entries)
    correct = sum(s["tp"] + s["tn"] for s in by_source.values())
    fp = sum(s["fp"] for s in by_source.values())
    fn = sum(s["fn"] for s in by_source.values())
    lo, hi = wilson(correct, n)
    lat.sort()

    print(f"\n=== {name} (n={n}) ===")
    print(f"accuracy {correct}/{n} = {correct/n:.1%}  Wilson95=[{lo:.1%},{hi:.1%}]")
    print(f"FPR={fp}/{fp+sum(s['tn'] for s in by_source.values())}"
          f"  FNR={fn}/{fn+sum(s['tp'] for s in by_source.values())}")
    print(f"latency ms: p50={pct(lat,50):.2f} p90={pct(lat,90):.2f} "
          f"p99={pct(lat,99):.2f} max={lat[-1]:.2f}")
    for src, s in sorted(by_source.items()):
        acc = (s["tp"] + s["tn"]) / s["n"]
        slo, shi = wilson(s["tp"] + s["tn"], s["n"])
        print(f"  {src:<18} n={s['n']:<5} acc={acc:.1%} [{slo:.1%},{shi:.1%}]  "
              f"TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']}")
    return {"n": n, "correct": correct, "wilson95": [round(lo, 4), round(hi, 4)],
            "by_source": by_source}


def main():
    results = {}
    hc3 = json.load(open(os.path.join(HERE, "data/benchmarks/hc3_bench.json")))
    pol = json.load(open(os.path.join(HERE, "data/benchmarks/polished_ai_bench.json")))
    orig = json.load(open(os.path.join(HERE, "data/test_dataset.json")))["entries"]
    for e in orig:
        e.setdefault("source", f"orig_{e['id'].rsplit('_',1)[0]}")

    results["hc3"] = run("HC3 (real-world ChatGPT vs human)", hc3)
    results["polished_ai_v7"] = run("Polished v7-edited AI essays (adversarial)", pol)
    results["original_slop_set"] = run("Original 55-entry slop set (in-distribution)", orig)

    all_entries = hc3 + pol + orig
    results["combined"] = run("COMBINED", all_entries)

    with open("experiments/results/noslop_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved -> noslop_benchmark_results.json")


if __name__ == "__main__":
    main()
