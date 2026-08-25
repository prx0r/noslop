"""Logistic-regression head over all NoSlop signals — binary AI classification.

Uses the same precomputed-signal trick as evolve_binary.py but adds the
unused features from features.py and trains a proper model instead of the
hand-designed linear scorer. Reports honest held-out numbers + Wilson CIs.

Run: python3 train_logreg.py
"""
import json
import os
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "miner"))
from miner.src import features as F
from miner.src.features import FEATURE_FNS, FEATURE_NAMES
from evolve_heldout import wilson

SEED = 42


def featurize(text: str) -> list[float]:
    return [fn(text) for fn in FEATURE_FNS.values()]


def report(name, y, pred):
    c = int(np.sum(pred == np.array(y)))
    lo, hi = wilson(c, len(y))
    tp = int(np.sum((pred == 1) & (np.array(y) == 1)))
    fp = int(np.sum((pred == 1) & (np.array(y) == 0)))
    fn = int(np.sum((pred == 0) & (np.array(y) == 1)))
    tn = int(np.sum((pred == 0) & (np.array(y) == 0)))
    print(f"{name:<22} {c}/{len(y)} = {c/len(y):.1%}  Wilson95=[{lo:.1%},{hi:.1%}]  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return {"correct": c, "n": len(y), "wilson95": [round(lo, 4), round(hi, 4)],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    hc3 = json.load(open("data/benchmarks/hc3_bench.json"))
    orig = json.load(open("data/test_dataset.json"))["entries"]
    semeval = json.load(open("data/benchmarks/semeval_bench.json"))
    rng = random.Random(SEED)
    humans = [e for e in hc3 if e["label"] == 0]
    ais = [e for e in hc3 if e["label"] == 1]
    rng.shuffle(humans)
    rng.shuffle(ais)

    sem_humans = sorted([e for e in semeval if e["label"] == 0], key=lambda e: hash(e["text"]) % 10**9)
    sem_ais = sorted([e for e in semeval if e["label"] == 1], key=lambda e: hash(e["text"]) % 10**9)
    rng2 = random.Random(SEED + 7)
    rng2.shuffle(sem_humans)
    rng2.shuffle(sem_ais)
    sem_train = sem_humans[:500] + sem_ais[:500]
    train_keys = {e["text"][:80] for e in sem_train}
    sem_eval = [e for e in semeval if e["text"][:80] not in train_keys]

    heldout = humans[500:900] + ais[500:900]
    y_ho = [e["label"] for e in heldout]

    print(f"featurizing {len(heldout)} heldout / {len(orig)} orig ...")
    X_ho = np.array([featurize(e["text"]) for e in heldout])
    X_orig = np.array([featurize(e["text"]) for e in orig])

    # Member 1: HC3 specialist (no SemEval data).
    sp_train = humans[:500] + ais[:500] + orig * 3
    X_train = np.array([featurize(e["text"]) for e in sp_train])
    y_train = [e["label"] for e in sp_train]
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X_train), y_train)

    # Member 2: multi-domain (adds the SemEval train slice).
    md_train = sp_train + sem_train
    X_md = np.array([featurize(e["text"]) for e in md_train])
    scaler_md = StandardScaler().fit(X_md)
    model_md = LogisticRegression(max_iter=2000, C=1.0).fit(
        scaler_md.transform(X_md), [e["label"] for e in md_train])

    print("\n=== ENSEMBLE logreg_v4 candidate (41 signals) ===")
    P = lambda m_, sc_, Xs: m_.predict_proba(sc_.transform(Xs))[:, 1]
    def ens_pred(Xs):
        return ((0.3 * P(model, scaler, Xs) + 0.7 * P(model_md, scaler_md, Xs)) >= 0.5).astype(int)
    r_ho = report("heldout HC3", y_ho, ens_pred(X_ho))
    r_orig = report("original set", [e['label'] for e in orig], ens_pred(X_orig))
    r_sem = report("SemEval OOD (unseen slice)",
                   [e['label'] for e in sem_eval],
                   ens_pred(np.array([featurize(e["text"]) for e in sem_eval])))

    coefs = sorted(zip(FEATURE_NAMES, (
        0.3 * model.coef_[0] + 0.7 * model_md.coef_[0])), key=lambda x: -abs(x[1]))
    print("\ntop blended coefficients:")
    for name, c in coefs[:8]:
        print(f"  {name:<14} {c:+.3f}")

    def member(m_, sc_, extra: dict) -> dict:
        return {
            "scaler_mean": sc_.mean_.tolist(),
            "scaler_scale": sc_.scale_.tolist(),
            "coef": m_.coef_[0].tolist(),
            "intercept": float(m_.intercept_[0]),
            **extra,
        }

    results = {
        "model": "logistic_regression_ensemble",
        "version": "logreg_v4_ensemble_41signals",
        "weights": [0.3, 0.7],
        "features": FEATURE_NAMES,
        "members": [
            member(model, scaler, {"id": "hc3_specialist",
                                   "train_spec": "hc3 500/500 + original55 x3"}),
            member(model_md, scaler_md, {"id": "multidomain",
                                         "train_spec": "hc3 500/500 + semeval 500/500 + original55 x3"}),
        ],
        "train_size": len(sp_train) + len(sem_train),
        "heldout_hc3": r_ho,
        "original_set": r_orig,
        "semeval_ood": r_sem,
    }
    # Tamper-evident manifest: hash datasets + params so any edit changes the id.
    import hashlib
    ds_hash = hashlib.sha256()
    for p in ("data/benchmarks/hc3_bench.json", "data/test_dataset.json",
              "data/benchmarks/semeval_bench.json"):
        ds_hash.update(open(p, "rb").read())
    manifest = {
        "datasets_sha256": ds_hash.hexdigest(),
        "model_sha256": hashlib.sha256(
            json.dumps({k: results[k] for k in
                        ("features", "weights", "members")},
                       sort_keys=True).encode()).hexdigest(),
        "seed": SEED,
        "train_spec": "member1 hc3-only; member2 +semeval 500/500 (multi-domain)",
        "holdout_spec": "hc3 humans[500:900]+ais[500:900]; semeval minus train slice",
    }
    results["manifest"] = manifest
    json.dump(results, open("experiments/results/noslop_logreg_model_training.json", "w"), indent=2)
    json.dump(results, open("miner/src/noslop_logreg_model.json", "w"), indent=2)

    # pure-python inference check (deployment has no numpy/sklearn)
    def infer_member(memb, feats):
        z = memb["intercept"] + sum(
            (f - mu) / s * c for f, mu, s, c in
            zip(feats, memb["scaler_mean"], memb["scaler_scale"], memb["coef"]))
        return 1.0 / (1.0 + pow(2.718281828459045, -z))

    preds_ens = []
    for row in X_ho:
        feats = row.tolist()
        p = 0.3 * infer_member(results["members"][0], feats) + \
            0.7 * infer_member(results["members"][1], feats)
        preds_ens.append(1 if p >= 0.5 else 0)
    c_pp = sum(1 for p, y in zip(preds_ens, y_ho) if p == y)
    lo, hi = wilson(c_pp, len(y_ho))
    print(f"\npure-python ensemble inference : {c_pp}/{len(y_ho)} = {c_pp/len(y_ho):.1%} "
          f"Wilson95=[{lo:.1%},{hi:.1%}]")
    print("saved -> noslop_logreg_model.json (+ miner/src/)")


if __name__ == "__main__":
    main()
