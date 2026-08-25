"""Calibration + human-pool FPR thresholding (peer-review R3).

Method follows the Waterloo generalizability study:
  - fit Platt scaling on a calibration slice untouched by training/thresholds
  - choose the decision threshold as the 99th percentile of PLATT-scaled
    scores on a LARGE human-only pool -> promises ~1% FPR on human text
  - report TPR at that promise on unseen test sets
  - store calibrator in the model artifact

Run: python3 calibrate.py
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "miner"))
from miner.src.features import FEATURE_FNS
from eval_metrics import featurize


def member_logit(memb, X):
    return ((X - np.array(memb["scaler_mean"])) / np.array(memb["scaler_scale"])
            @ memb["coef"] + memb["intercept"])


def main():
    art = json.load(open("miner/src/noslop_logreg_model.json"))
    w = art.get("serving_params", {}).get("ensemble_weight_specialist", 0.25)
    members = art["members"]
    wv = np.array([w, 1 - w])

    def ens_logit(X):
        logits = np.vstack([member_logit(m, X) for m in members])
        return wv @ logits

    hc3 = json.load(open("data/benchmarks/hc3_bench.json"))
    sem = json.load(open("data/benchmarks/semeval_bench.json"))
    rng = __import__("random").Random(42)
    humans = [e for e in hc3 if e["label"] == 0]
    ais = [e for e in hc3 if e["label"] == 1]
    rng.shuffle(humans); rng.shuffle(ais)

    print("featurizing pools...")
    cal_texts = ([e["text"] for e in humans[1300:1800]] +
                 [e["text"] for e in ais[1300:1800]])
    y_cal = np.array([0]*500 + [1]*500)
    X_cal = featurize(cal_texts)
    L_cal = ens_logit(X_cal).reshape(-1, 1)

    # human-only pool for FPR promise (never used anywhere else)
    hu_texts = [e["text"] for e in humans[1800:]]
    y_hu = np.zeros(len(hu_texts), dtype=int)
    X_hu = featurize(hu_texts)
    L_hu = ens_logit(X_hu).reshape(-1, 1)

    # Platt: logistic on the raw ensemble logit
    platt = LogisticRegression().fit(L_cal, y_cal)

    def platt_p(L):
        return platt.predict_proba(L)[:, 1]

    # promised-FPR threshold from the human-only pool (99th pct of human scores)
    p_hu = platt_p(L_hu)
    tau = float(np.quantile(p_hu, 0.99))
    actual_fpr = float((p_hu >= tau).mean())

    # evaluate the promise on unseen AI test sets
    ho_ai_texts = [e["text"] for e in ais[500:900]]
    P_ai_ho = platt_p(ens_logit(featurize(ho_ai_texts)).reshape(-1, 1))
    tpr_ho = float((P_ai_ho >= tau).mean())

    sem_ai = [e["text"] for e in sem if e["label"] == 1][:800]
    P_ai_sem = platt_p(ens_logit(featurize(sem_ai)).reshape(-1, 1))
    tpr_sem = float((P_ai_sem >= tau).mean())

    # sanity: human holdout not in pool
    hu_hold = [e["text"] for e in humans[500:900]]
    p_hold = platt_p(ens_logit(featurize(hu_hold)).reshape(-1, 1))
    fpr_hold = float((p_hold >= tau).mean())

    brier_pre = float(np.mean(( (wv @ np.vstack([
        1/(1+np.exp(-member_logit(members[i], X_c))) for i, X_c in
        [(0, None)]*2 ])[0]) - y_cal)**2)) if False else None

    print(f"platt params: a={float(platt.coef_[0][0]):.4f} b={float(platt.intercept_[0]):+.4f}")
    print(f"human pool n={len(hu_texts)}: FPR at tau={tau:.4f} -> {actual_fpr:.2%}")
    print(f"human holdout n=400: FPR -> {fpr_hold:.2%}")
    print(f"TPR@promised-1%FPR: hc3 heldout AI={tpr_ho:.1%}, semeval AI(n=800)={tpr_sem:.1%}")

    art["calibration"] = {
        "method": "platt_on_ensemble_logit",
        "platt_coef": float(platt.coef_[0][0]),
        "platt_intercept": float(plat_intercept := float(platt.intercept_[0])),
        "fpr_promise": 0.01,
        "decision_threshold": tau,
        "threshold_source": "99th percentile of platt scores on 1785-doc human-only pool (humans[1800:])",
        "tpr_at_promise": {"hc3_heldout": round(tpr_ho, 4),
                           "semeval_800": round(tpr_sem, 4)},
        "human_holdout_fpr": round(fpr_hold, 4),
    }
    json.dump(art, open("noslop_logreg_model.json", "w"), indent=2)
    json.dump(art, open("miner/src/noslop_logreg_model.json", "w"), indent=2)
    print("saved -> artifact calibration block")


if __name__ == "__main__":
    main()
