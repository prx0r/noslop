"""Cogym-driven optimization of ensemble weight + decision threshold.

Uses cogym_kernel machinery:
  - evo.recipes (elitist_mutation via EvolutionContext / propose_children)
  - eval.gates  (QualityGate fail-closed promotion + wilson)

Data discipline:
  TRAIN   (model members): hc3[:500]+sem[:500]+orig x3   -- frozen, untouched here
  VAL     (optimizer sees): hc3[900:1300] + sem next 300/300
  TEST    (report only):    hc3[500:900] + rest of semeval + polished_ai

Run: python3 cogym_optimize.py
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, os_path := __file__.rsplit("/", 1)[0])
sys.path.insert(0, os_path + "/miner")
from miner.src.features import FEATURE_FNS

sys.path.insert(0, "repos/cogym")
from cogym_kernel.evo.recipes import EvolutionContext, propose_children
from cogym_kernel.eval.gates import QualityGate, gates_pass, wilson

SEED = 42
GENERATIONS = 40
POPULATION = 16
ELITE_K = 5

SEARCH_SPACE = {
    "w_specialist": {"min": 0, "max": 20, "step": 1},      # x0.05 -> 0.00..1.00
    "threshold": {"min": 30, "max": 70, "step": 2},        # x0.01 -> 0.30..0.70
}

GATES = [
    QualityGate(metric="acc_hc3_val", mode="max", value=0.75),
    QualityGate(metric="acc_sem_val", mode="max", value=0.75),
]


def featurize_all(entries):
    return np.array([[fn(e["text"]) for fn in FEATURE_FNS.values()] for e in entries])


class Ensemble:
    def __init__(self, path="noslop_logreg_model.json"):
        m = json.load(open(path))
        self.members = m["members"]
        self.manifest = m.get("manifest", {})

    def probs(self, X):
        out = []
        for memb in self.members:
            z = memb["intercept"] + (X - np.array(memb["scaler_mean"])) / np.array(memb["scaler_scale"]) @ np.array(memb["coef"])
            out.append(1.0 / (1.0 + np.exp(-z)))
        return np.vstack(out)


def metrics_for(cfg, P, ys):
    w = cfg["w_specialist"] * 0.05
    t = cfg["threshold"] * 0.01
    p = w * P[0] + (1 - w) * P[1]
    pred = (p >= t).astype(int)
    accs = {}
    for name, y in ys.items():
        accs[f"acc_{name}_val"] = float((pred[name] == y[name]).mean())
    accs["macro"] = sum(accs.values()) / len(accs)
    return accs


def main():
    ens = Ensemble()
    hc3 = json.load(open("data/benchmarks/hc3_bench.json"))
    sem = json.load(open("data/benchmarks/semeval_bench.json"))
    pol = json.load(open("data/benchmarks/polished_ai_bench.json"))
    rng = random.Random(SEED)
    humans = [e for e in hc3 if e["label"] == 0]
    ais = [e for e in hc3 if e["label"] == 1]
    rng.shuffle(humans); rng.shuffle(ais)

    sh = sorted([e for e in sem if e["label"] == 0], key=lambda e: hash(e["text"]) % 10**9)
    sa = sorted([e for e in sem if e["label"] == 1], key=lambda e: hash(e["text"]) % 10**9)
    r2 = random.Random(SEED + 7); r2.shuffle(sh); r2.shuffle(sa)

    val_h = humans[900:1300]; val_a = ais[900:1300]
    sem_tr_keys = {e["text"][:80] for e in sh[:500] + sa[:500]}
    sem_rest = [e for e in sem if e["text"][:80] not in sem_tr_keys]
    sem_rest_h = [e for e in sem_rest if e["label"] == 0][:300]
    sem_rest_a = [e for e in sem_rest if e["label"] == 1][:300]

    print("featurizing...")
    val_sets = {
        "hc3": val_h + val_a,
        "sem": sem_rest_h + sem_rest_a,
    }
    test_sets = {
        "hc3_test": humans[500:900] + ais[500:900],
        "sem_test": [e for e in sem_rest
                     if e["text"][:80] not in {x['text'][:80] for x in sem_rest_h + sem_rest_a}],
        "polished_ai": pol,
    }
    PV, PT = {}, {}
    YV, YT = {}, {}
    for k, es in val_sets.items():
        PV[k] = ens.probs(featurize_all(es))
        YV[k] = np.array([e["label"] for e in es])
    for k, es in test_sets.items():
        PT[k] = ens.probs(featurize_all(es))
        YT[k] = np.array([e["label"] for e in es])

    def evaluate(cfg, P, ys):
        w = cfg["w_specialist"] * 0.05
        t = cfg["threshold"] * 0.01
        p = w * P[list(P)[0]] + (1 - w) * P[list(P)[1]]
        out = {}
        idx = {name: i for i, name in enumerate(P)}
        for name in P:
            pi = w * P[name][0] + (1 - w) * P[name][1]
            out[name] = ((pi >= t).astype(int) == ys[name]).mean()
        return out

    def metrics(cfg):
        m = {}
        for name in val_sets:
            w = cfg["w_specialist"] * 0.05
            t = cfg["threshold"] * 0.01
            pi = w * PV[name][0] + (1 - w) * PV[name][1]
            m[f"acc_{name}_val"] = float(((pi >= t).astype(int) == YV[name]).mean())
        m["macro"] = sum(m.values()) / len([n for n in m if n.startswith("acc")])
        return m

    rng_opt = random.Random(SEED + 99)
    population = [dict(w_specialist=rng_opt.randint(0, 20),
                       threshold=rng_opt.randint(30, 70)) for _ in range(POPULATION)]
    elites, scorecard = [], []
    best = None
    for gen in range(GENERATIONS):
        scored = []
        for cfg in population:
            m = metrics(cfg)
            ok = gates_pass(GATES, m)
            scored.append((m, cfg, ok))
            scorecard.append({"config": cfg, "metrics": m})
        passing = [s for s in scored if s[2]]
        pool = passing if passing else scored
        pool.sort(key=lambda s: (-s[0]["macro"],))
        gen_best = pool[0]
        if best is None or gen_best[0]["macro"] > best[0]["macro"]:
            best = gen_best
        ctx = EvolutionContext(
            elite_configs=[c for _, c, _ in pool[:ELITE_K]],
            scorecard=scorecard[-100:],
            hydra_leaders=[], search_space=SEARCH_SPACE, rng=rng_opt)
        children = propose_children("elitist_mutation", ctx, POPULATION - ELITE_K)
        population = [c for _, c, _ in pool[:ELITE_K]] + children
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            gk = "PASS" if gen_best[2] else "FAIL"
            print(f"gen {gen:3d}: macro={gen_best[0]['macro']:.4f} [{gk}] "
                  f"w={gen_best[1]['w_specialist']*0.05:.2f} t={gen_best[1]['threshold']*0.01:.2f}")

    wm, cfg = best[1], dict(best[1])
    w = cfg["w_specialist"] * 0.05
    t = cfg["threshold"] * 0.01
    print(f"\n=== CHAMPION === w_specialist={w:.2f} threshold={t:.2f} "
          f"val_macro={best[0]['macro']:.4f}")

    print("\n=== UNTOUCHED TEST SETS ===")
    test_metrics = {}
    for name in test_sets:
        pi = w * PT[name][0] + (1 - w) * PT[name][1]
        pred = (pi >= t).astype(int)
        c = int((pred == YT[name]).sum()); n = len(YT[name])
        wr = wilson(c, n)
        test_metrics[name] = wr
        print(f"{name:<12} {c}/{n} = {wr['p']:.1%}  Wilson95=[{wr['lo']:.1%},{wr['hi']:.1%}]")

    # Write serving params back into the model artifact.
    art = json.load(open("noslop_logreg_model.json"))
    art["serving_params"] = {
        "ensemble_weight_specialist": w,
        "decision_threshold": t,
        "optimized_by": "cogym_kernel elitist_mutation",
        "val_spec": "hc3[900:1300] + semeval 300/300 (never trained)",
        "gates": "acc>=0.75 both val slices",
    }
    json.dump(art, open("noslop_logreg_model.json", "w"), indent=2)
    json.dump(art, open("miner/src/noslop_logreg_model.json", "w"), indent=2)
    json.dump({"champion_config": cfg, "val_metrics": wm,
               "test_metrics": test_metrics},
              open("experiments/results/noslop_cogym_optimize_results.json", "w"), indent=2)
    print("\nsaved -> noslop_cogym_optimize_results.json (+ serving_params in model)")


if __name__ == "__main__":
    main()
