"""Binary-classification weight evolution against the large HC3 benchmark.

Raw signals are precomputed once per text; the GA then evaluates configs with
pure arithmetic (identical formula to miner.src.features.extract), making
large-population evolution fast.

Train: 500/500 balanced HC3 + original 55-entry set (x3 weight)
Held-out: 400/400 balanced HC3, never seen during search
Run: python3 evolve_binary.py
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from miner.src import features as F
from evolve_heldout import SEARCH_SPACE, wilson, sample_config, mutate

SEED = 42
GENERATIONS = 300
POPULATION_SIZE = 24
ELITE_K = 8


def raw_signals(text: str) -> dict:
    """All raw quantities the scoring formula needs, computed once."""
    pats = F.detect_patterns(text)
    n_sent = F.count_sentences(text)
    return {
        "narr": pats["NARR"], "neg": pats["NEG"],
        "three_list": pats["3LIST"], "cliche": pats["CLICHE"],
        "n_sent": n_sent,
        "burs": F.burstiness(text),
        "punct": F.punctuation_density(text),
        "ld": F.ld_score(text),
        "ngram_dens": F.ngram_tell_density(text),
        "the_r": F.the_rate(text),
        "abs_noun": F.abstract_noun_density(text),
        "commitment": F.voice_commitment(text),
        "staging": F.staging_count(text),
        "lineage": F.lineage_chains(text),
    }


def predict(sig: dict, w: dict, ai_threshold: float) -> int:
    total_patterns = (
        sig["narr"] * w["narr"] + sig["neg"] * w["neg"] +
        sig["three_list"] * w["three_list"] + sig["cliche"] * w["cliche"])
    pattern_rate = total_patterns / sig["n_sent"] if sig["n_sent"] else 0

    score = 0.0
    score += min(pattern_rate / w["pattern_threshold"], 1.0) * 0.25
    score += max(0.0, 1.0 - sig["ld"] / w["ld_threshold"]) * 0.15
    score += min(sig["ngram_dens"] / w["ngram_threshold"], 1.0) * 0.10
    score += max(0.0, 1.0 - sig["burs"] / w["burstiness_threshold"]) * 0.10
    score += max(0.0, min(1.0, sig["the_r"] / w["the_threshold"])) * 0.10
    score += max(0.0, min(1.0, sig["abs_noun"] / w["abstract_threshold"])) * 0.10
    score += max(0.0, 1.0 - sig["commitment"] / w["commitment_threshold"]) * 0.10
    score += max(0.0, 1.0 - sig["punct"] / w["punct_threshold"]) * 0.05
    score += min(1.0, (sig["staging"] + sig["lineage"]) / 3) * 0.05
    confidence = max(0.05, min(0.99, score))
    return 1 if confidence >= ai_threshold else 0


def accuracy(sigs, labels, cfg):
    w = {k: cfg[f"{k}_weight"] if not k.endswith("_threshold") else cfg[k]
         for k in ("narr", "neg", "three_list", "cliche", "pattern_threshold",
                   "burstiness_threshold", "punct_threshold", "ld_threshold",
                   "ngram_threshold", "the_threshold", "abstract_threshold",
                   "commitment_threshold")}
    t = cfg["ai_threshold"]
    c = sum(1 for s, y in zip(sigs, labels) if predict(s, w, t) == y)
    return c


def main():
    hc3 = json.load(open("data/benchmarks/hc3_bench.json"))
    orig = json.load(open("data/test_dataset.json"))["entries"]
    rng = random.Random(SEED)

    humans = [e for e in hc3 if e["label"] == 0]
    ais = [e for e in hc3 if e["label"] == 1]
    rng.shuffle(humans)
    rng.shuffle(ais)

    train_hc3 = [(e, 0) for e in humans[:500]] + [(e, 1) for e in ais[:500]]
    ho_hc3 = [(e, 0) for e in humans[500:900]] + [(e, 1) for e in ais[500:900]]
    train_all = train_hc3 + [(e, e["label"]) for e in orig] * 3

    print(f"precomputing signals: {len(train_all)} train, {len(ho_hc3)} heldout...")
    tr_sigs = [raw_signals(e["text"]) for e, _ in train_all]
    tr_labels = [y for _, y in train_all]
    ho_sigs = [raw_signals(e["text"]) for e, _ in ho_hc3]
    ho_labels = [y for _, y in ho_hc3]
    orig_sigs = [raw_signals(e["text"]) for e in orig]
    orig_labels = [e["label"] for e in orig]

    population = [sample_config(rng) for _ in range(POPULATION_SIZE)]
    best_ever, best_cfg = -1.0, None
    for gen in range(GENERATIONS):
        scored = sorted(((accuracy(tr_sigs, tr_labels, cfg) / len(tr_labels), cfg)
                         for cfg in population), key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_ever:
            best_ever, best_cfg = scored[0][0], scored[0][1]
        if gen % 50 == 0 or gen == GENERATIONS - 1:
            print(f"gen {gen:3d}: train best={best_ever:.4f} gen_best={scored[0][0]:.4f}")
        elites = [cfg for _, cfg in scored[:ELITE_K]]
        children = [mutate(elites[n % len(elites)], rng)
                    for n in range(POPULATION_SIZE - ELITE_K)]
        population = elites + children

    def eval_set(sigs, labels, cfg):
        c = accuracy(sigs, labels, cfg)
        lo, hi = wilson(c, len(labels))
        return c, len(labels), lo, hi

    print("\n=== NEW WEIGHTS ===")
    c, n, lo, hi = eval_set(ho_sigs, ho_labels, best_cfg)
    print(f"heldout HC3 : {c}/{n} = {c/n:.1%}  Wilson95=[{lo:.1%},{hi:.1%}]")
    co, no, loo, hio = eval_set(orig_sigs, orig_labels, best_cfg)
    print(f"original set: {co}/{no} = {co/no:.1%}  Wilson95=[{loo:.1%},{hio:.1%}]")

    old_cfg = {"narr_weight": 1.0, "neg_weight": 0.3, "three_list_weight": 0.2,
               "cliche_weight": 5.0, "pattern_threshold": 1.0,
               "burstiness_threshold": 0.3, "punct_threshold": 0.03,
               "ld_threshold": 0.4, "ngram_threshold": 0.3,
               "fw_threshold": 0.5, "the_threshold": 0.2,
               "abstract_threshold": 0.8, "commitment_threshold": 0.3,
               "ai_threshold": 0.35}
    c2, n2, lo2, hi2 = eval_set(ho_sigs, ho_labels, old_cfg)
    print(f"\n=== OLD WEIGHTS (for reference) ===")
    print(f"heldout HC3 : {c2}/{n2} = {c2/n2:.1%}  Wilson95=[{lo2:.1%},{hi2:.1%}]")
    c2o, n2o, loo2, hio2 = eval_set(orig_sigs, orig_labels, old_cfg)
    print(f"original set: {c2o}/{n2o} = {c2o/n2o:.1%}")

    json.dump({"winner_config": best_cfg,
               "train_acc": best_ever,
               "heldout": {"correct": c, "n": n, "wilson95": [round(lo, 4), round(hi, 4)]},
               "original_set": {"correct": co, "n": no},
               "old_weights_heldout": {"correct": c2, "n": n2},
               "generations": GENERATIONS, "population": POPULATION_SIZE,
               "train_size": len(tr_labels)},
              open("noslop_binary_evolution_results.json", "w"), indent=2)
    print("\nsaved -> noslop_binary_evolution_results.json")


if __name__ == "__main__":
    main()
