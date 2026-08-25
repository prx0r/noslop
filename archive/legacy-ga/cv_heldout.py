"""Stratified 5-fold CV of the full noslop procedure (GA + features).

Each fold: evolve weights on 4 folds, evaluate on the held-out fold.
Gives a tighter honest estimate than the single 60/40 split.

Run: python3 cv_heldout.py
"""
import json
import math
import random
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from evolve_heldout import (SEARCH_SPACE, GENERATIONS, POPULATION_SIZE, ELITE_K,
                            SEED, wilson, to_weights, accuracy,
                            sample_config, mutate)


def stratified_folds(entries, k, rng):
    folds = [[] for _ in range(k)]
    for label in (0, 1):
        group = sorted([e for e in entries if e["label"] == label], key=lambda e: e["id"])
        rng.shuffle(group)
        for i, e in enumerate(group):
            folds[i % k].append(e)
    return folds


def evolve_on(train, rng, generations=GENERATIONS):
    population = [sample_config(rng) for _ in range(POPULATION_SIZE)]
    best_ever, best_cfg = -1.0, None
    for gen in range(generations):
        scored = []
        for cfg in population:
            w = to_weights(cfg)
            c, _ = accuracy(train, w, cfg["ai_threshold"])
            scored.append((c / len(train), cfg))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_ever:
            best_ever, best_cfg = scored[0][0], scored[0][1]
        elites = [cfg for _, cfg in scored[:ELITE_K]]
        children = [mutate(elites[n % len(elites)], rng) for n in range(POPULATION_SIZE - ELITE_K)]
        population = elites + children
    return best_cfg, best_ever


def main():
    with open("data/test_dataset.json") as f:
        entries = json.load(f)["entries"]

    K = 5
    rng = random.Random(SEED + 1)
    folds = stratified_folds(entries, K, rng)

    correct_total, n_total = 0, 0
    fold_results = []
    for i in range(K):
        test = folds[i]
        train = [e for j in range(K) if j != i for e in folds[j]]
        fold_rng = random.Random(SEED + 100 + i)
        # faster convergence is fine: GA plateaus by gen ~20
        cfg, tr_acc = evolve_on(train, fold_rng, generations=60)
        c, misses = accuracy(test, to_weights(cfg), cfg["ai_threshold"])
        lo, hi = wilson(c, len(test))
        print(f"fold {i}: train={len(train)} test={len(test)} "
              f"train_acc={tr_acc:.1%} test={c}/{len(test)} ({c/len(test):.0%}) "
              f"misses={[m['id'] for m in misses]}")
        fold_results.append({"fold": i, "train_size": len(train),
                             "test_size": len(test), "correct": c,
                             "config": cfg, "misses": misses})
        correct_total += c
        n_total += len(test)

    lo, hi = wilson(correct_total, n_total)
    print(f"\nPOOLED: {correct_total}/{n_total} = {correct_total/n_total:.1%}  "
          f"Wilson95=[{lo:.1%}, {hi:.1%}]")

    with open("noslop_cv_results.json", "w") as f:
        json.dump({"pooled_correct": correct_total, "pooled_n": n_total,
                   "wilson95": [round(lo, 4), round(hi, 4)],
                   "folds": fold_results}, f, indent=2)
    print("saved -> noslop_cv_results.json")


if __name__ == "__main__":
    main()
