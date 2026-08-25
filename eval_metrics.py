"""Frontier-standard metric suite for NoSlop (peer-review R1, R2, R4).

Implements the five-metric convention from arXiv:2603.17522 (AUROC, AUPRC,
Brier, EER-thresholded accuracy, TPR@FPR-1%) plus length-bucketed accuracy,
per-member permutation importance, and per-member coefficient inspection.

Run: python3 eval_metrics.py
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "miner"))
from miner.src.features import FEATURE_FNS, FEATURE_NAMES
from evolve_heldout import wilson


def featurize(texts):
    return np.array([[fn(t) for fn in FEATURE_FNS.values()] for t in texts])


def eer_threshold(y, p):
    """Threshold where FPR == FNR."""
    thresholds = np.unique(np.round(p, 4))
    best_t, best_diff = 0.5, 1e9
    for t in thresholds:
        pred = p >= t
        fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        fnr = ((~pred) & (y == 1)).sum() / max(1, (y == 1).sum())
        if abs(fpr - fnr) < best_diff:
            best_diff, best_t = abs(fpr - fnr), t
    return float(best_t)


def five_metrics(y, p):
    y = np.array(y)
    pred05 = (p >= 0.5).astype(int)
    t_eer = eer_threshold(y, p)
    pred_eer = (p >= t_eer).astype(int)
    fpr1 = (p[p.argsort()] >= np.quantile(p[y == 0], 0.99))  # placeholder replaced below
    # TPR @ FPR<=1%: threshold = 99th percentile of HUMAN scores
    tau = np.quantile(p[y == 0], 0.99)
    tpr_at_fpr1 = float((p[y == 1] >= tau).mean())
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "acc@eer_thr": round(float((pred_eer == y).mean()), 4),
        "eer_threshold": round(t_eer, 4),
        "tpr@fpr1%": round(tpr_at_fpr1, 4),
    }


def length_buckets(texts, y, p):
    lens = np.array([len(t.split()) for t in texts])
    out = {}
    for lo, hi in [(50, 150), (150, 300), (300, 10_000)]:
        m = (lens >= lo) & (lens < hi)
        if m.sum() < 20:
            continue
        acc = float(((p[m] >= 0.5).astype(int) == y[m]).mean())
        out[f"{lo}-{hi}w"] = {"n": int(m.sum()), "acc": round(acc, 4)}
    short = lens < 50
    if short.sum():
        out["<50w"] = {"n": int(short.sum()),
                       "acc": round(float(((p[short] >= 0.5).astype(int) == y[short]).mean()), 4)}
    return out


def permutation_importance(X, y, memb, names, n_rep=3, seed=0):
    """Per-member importance: shuffle one feature, measure prob shift."""
    rng = np.random.default_rng(seed)
    mu, sd, coef, b = (np.array(memb["scaler_mean"]), np.array(memb["scaler_scale"]),
                       np.array(memb["coef"]), memb["intercept"])
    base_p = 1 / (1 + np.exp(-(X - mu) / sd @ coef + b))
    base_acc = ((base_p >= 0.5).astype(int) == y).mean()
    out = {}
    for j, name in enumerate(names):
        deltas = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pp = 1 / (1 + np.exp(-(Xp - mu) / sd @ coef + b))
            deltas.append(base_acc - ((pp >= 0.5).astype(int) == y).mean())
        out[name] = round(float(np.mean(deltas)), 5)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    art = json.load(open("miner/src/noslop_logreg_model.json"))
    hc3 = json.load(open("data/benchmarks/hc3_bench.json"))
    sem = json.load(open("data/benchmarks/semeval_bench.json"))
    rng = __import__("random").Random(42)
    humans = [e for e in hc3 if e["label"] == 0]; ais = [e for e in hc3 if e["label"] == 1]
    rng.shuffle(humans); rng.shuffle(ais)

    ho_texts = ([e["text"] for e in humans[500:900]] +
                [e["text"] for e in ais[500:900]])
    y_ho = np.array([0] * 400 + [1] * 400)

    print("featurizing held-out...")
    X = featurize(ho_texts)

    def member_prob(memb, Xs):
        z = (Xs - memb["scaler_mean"]) / memb["scaler_scale"] @ memb["coef"] + memb["intercept"]
        return 1 / (1 + np.exp(-z))

    w = art.get("serving_params", {}).get("ensemble_weight_specialist", 0.25)
    P = (w * member_prob(art["members"][0], X) +
         (1 - w) * member_prob(art["members"][1], X))

    print("\n=== FIVE-METRIC SUITE (heldout HC3, n=800) ===")
    print(json.dumps(five_metrics(y_ho, P), indent=1))

    print("\n=== LENGTH BUCKETS ===")
    print(json.dumps(length_buckets(ho_texts, y_ho, P), indent=1))

    print("\n=== PER-MEMBER COEFFICIENT SIGN AUDIT: 'neg' feature ===")
    for i, memb in enumerate(art["members"]):
        c = memb["coef"][FEATURE_NAMES.index("neg")]
        print(f"  member[{i}] ({memb.get('id','?')}): neg coef = {c:+.4f}")

    print("\n=== PERMUTATION IMPORTANCE (top 8, per member) ===")
    for i, memb in enumerate(art["members"]):
        imp = permutation_importance(X, y_ho, memb, FEATURE_NAMES)
        print(f"member[{i}] {memb.get('id','?')}:")
        for k in list(imp)[:8]:
            print(f"   {k:<16} {imp[k]:+.4f}")

    json.dump({
        "five_metrics_heldout_hc3": five_metrics(y_ho, P),
        "length_buckets": length_buckets(ho_texts, y_ho, P),
        "neg_coefficients": {
            f"member_{m['members'][i].get('id', i)}":
                m["members"][i]["coef"][FEATURE_NAMES.index("neg")]
            for i, m in [(0, art), (1, art)]},
    }, open("experiments/results/noslop_frontier_metrics.json", "w"), indent=2)
    print("\nsaved -> noslop_frontier_metrics.json")


if __name__ == "__main__":
    main()
