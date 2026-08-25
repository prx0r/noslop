"""Honest generalization eval for noslop weights.

Procedure:
  1. Evolve classifier weights using ONLY the train split (same GA as v2).
  2. Evaluate on held-out split — never seen by evolution.
  3. Compare against the original v2 config (which was tuned on all 55
     entries, including the held-out ones — expected optimistically biased).
  4. Wilson score intervals on all proportions.

Run: python3 evolve_heldout.py
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from miner.src.features import extract

SEED = 42
GENERATIONS = 150
POPULATION_SIZE = 20
ELITE_K = 6

SEARCH_SPACE = {
    "narr_weight": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    "neg_weight": [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    "three_list_weight": [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    "cliche_weight": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "pattern_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    "burstiness_threshold": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
    "punct_threshold": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    "ld_threshold": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    "ngram_threshold": [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    "fw_threshold": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    "the_threshold": [0.2, 0.25, 0.30, 0.35, 0.40, 0.50],
    "abstract_threshold": [0.4, 0.5, 0.6, 0.7, 0.8],
    "commitment_threshold": [0.05, 0.1, 0.15, 0.2, 0.3],
    "ai_threshold": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
}

V2_WEIGHTS = {
    "narr": 1.0, "neg": 0.3, "three_list": 0.2, "cliche": 5.0,
    "pattern_threshold": 1.0, "burstiness_threshold": 0.3,
    "punct_threshold": 0.03, "ld_threshold": 0.4,
    "ngram_threshold": 0.3, "fw_threshold": 0.5,
    "the_threshold": 0.2, "abstract_threshold": 0.8,
    "commitment_threshold": 0.3,
}
V2_AI_THRESHOLD = 0.35


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def to_weights(cfg: dict) -> dict:
    return {
        "narr": cfg["narr_weight"], "neg": cfg["neg_weight"],
        "three_list": cfg["three_list_weight"], "cliche": cfg["cliche_weight"],
        "pattern_threshold": cfg["pattern_threshold"],
        "burstiness_threshold": cfg["burstiness_threshold"],
        "punct_threshold": cfg["punct_threshold"],
        "ld_threshold": cfg["ld_threshold"],
        "ngram_threshold": cfg["ngram_threshold"],
        "fw_threshold": cfg["fw_threshold"],
        "the_threshold": cfg["the_threshold"],
        "abstract_threshold": cfg["abstract_threshold"],
        "commitment_threshold": cfg["commitment_threshold"],
    }


def accuracy(entries: list[dict], weights: dict, ai_threshold: float) -> tuple[int, list]:
    correct, misses = 0, []
    for e in entries:
        fv = extract(e["text"], weights)
        pred = 1 if fv.confidence >= ai_threshold else 0
        if pred == e["label"]:
            correct += 1
        else:
            misses.append({"id": e["id"], "true": e["label"],
                           "conf": round(fv.confidence, 3)})
    return correct, misses


def sample_config(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def mutate(parent: dict, rng: random.Random, rate: float = 0.3) -> dict:
    child = dict(parent)
    for k in SEARCH_SPACE:
        if rng.random() < rate:
            child[k] = rng.choice(SEARCH_SPACE[k])
    return child


def main():
    with open("data/train_split.json") as f:
        train = json.load(f)["entries"]
    with open("data/heldout_split.json") as f:
        heldout = json.load(f)["entries"]

    print(f"train={len(train)} heldout={len(heldout)}")
    print(f"evolving on TRAIN ONLY: {GENERATIONS} gens x pop {POPULATION_SIZE}")
    print()

    rng = random.Random(SEED)
    population = [sample_config(rng) for _ in range(POPULATION_SIZE)]
    best_ever, best_cfg = -1.0, None

    for gen in range(GENERATIONS):
        scored = []
        for cfg in population:
            w = to_weights(cfg)
            t = cfg["ai_threshold"]
            c, _ = accuracy(train, w, t)
            scored.append((c / len(train), cfg))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_ever:
            best_ever = scored[0][0]
            best_cfg = scored[0][1]

        if gen % 20 == 0 or gen == GENERATIONS - 1:
            avg = sum(s for s, _ in scored) / len(scored)
            print(f"gen {gen:3d}: train avg={avg:.3f} best={scored[0][0]:.3f} "
                  f"best_ever={best_ever:.3f}")

        elites = [cfg for _, cfg in scored[:ELITE_K]]
        children = [mutate(elites[i % len(elites)], rng) for i in range(POPULATION_SIZE - ELITE_K)]
        population = elites + children

    print("\n=== EVALUATION ===")

    clean_w, clean_t = to_weights(best_cfg), best_cfg["ai_threshold"]
    tr_c, _ = accuracy(train, clean_w, clean_t)
    ho_c, ho_misses_clean = accuracy(heldout, clean_w, clean_t)
    lo, hi = wilson(ho_c, len(heldout))
    print(f"[clean procedure]  train={tr_c}/{len(train)} ({tr_c/len(train):.1%})  "
          f"heldout={ho_c}/{len(heldout)} ({ho_c/len(heldout):.1%})  "
          f"Wilson95=[{lo:.1%},{hi:.1%}]")
    print(f"  config: {json.dumps(best_cfg)}")
    for m in ho_misses_clean:
        print(f"  MISS {m['id']}: true={m['true']} conf={m['conf']}")

    v2_w, v2_t = V2_WEIGHTS, V2_AI_THRESHOLD
    tr_c2, _ = accuracy(train, v2_w, v2_t)
    ho_c2, ho_misses_v2 = accuracy(heldout, v2_w, v2_t)
    lo2, hi2 = wilson(ho_c2, len(heldout))
    print(f"\n[v2 tuned-on-all]  train={tr_c2}/{len(train)} ({tr_c2/len(train):.1%})  "
          f"heldout={ho_c2}/{len(heldout)} ({ho_c2/len(heldout):.1%})  "
          f"Wilson95=[{lo2:.1%},{hi2:.1%}]")
    for m in ho_misses_v2:
        print(f"  MISS {m['id']}: true={m['true']} conf={m['conf']}")

    results = {
        "split_manifest": "data/heldout_split.json",
        "procedure": "GA on train split only; held-out never touched during search",
        "generations": GENERATIONS,
        "population_size": POPULATION_SIZE,
        "clean_procedure": {
            "config": best_cfg,
            "train_correct": tr_c, "train_size": len(train),
            "heldout_correct": ho_c, "heldout_size": len(heldout),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "misses": ho_misses_clean,
        },
        "v2_tuned_on_all": {
            "heldout_correct": ho_c2, "heldout_size": len(heldout),
            "wilson95": [round(lo2, 4), round(hi2, 4)],
            "misses": ho_misses_v2,
        },
    }
    with open("noslop_heldout_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved -> noslop_heldout_results.json")


if __name__ == "__main__":
    main()
