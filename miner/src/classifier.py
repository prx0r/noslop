"""NoSlop ensemble classifier over v6+stylometric signals.

Deployment-safe: pure-python inference from a frozen JSON artifact
(noslop_logreg_model.json), no numpy/sklearn required at serve time.

logreg_v3: weighted ensemble of two logistic members
  - hc3_specialist (x0.3): trained on HC3 + original set
  - multidomain     (x0.7): + SemEval sample for domain robustness
Signals: v6 slop patterns + excess-vocab focal words + stylometrics (24).
"""
import json
import math
import os

try:
    from features import FEATURE_FNS, FEATURE_NAMES
except ImportError:
    from miner.src.features import FEATURE_FNS, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "noslop_logreg_model.json")


class SlopClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        with open(model_path) as f:
            m = json.load(f)
        self.version = m.get("version", m.get("model", "?"))
        self.weights = m.get("weights")
        if self.weights is None:
            self.members = [{"scaler_mean": m["scaler_mean"],
                             "scaler_scale": m["scaler_scale"],
                             "coef": m["coef"],
                             "intercept": m["intercept"]}]
            self.weights = [1.0]
        else:
            self.members = m["members"]
        self.manifest = m.get("manifest", {})
        self.calibration = m.get("calibration", {})
        sp = m.get("serving_params", {})
        self.threshold = self.calibration.get(
            "decision_threshold", sp.get("decision_threshold", 0.5))
        w = sp.get("ensemble_weight_specialist")
        if w is not None and len(self.weights) == 2:
            self.weights = [w, 1.0 - w]

    def probability(self, feats: list[float]) -> float:
        # weighted sum of member LOGITS (same space the Platt calibrator was fit in)
        zl = 0.0
        for wt, memb in zip(self.weights, self.members):
            z = memb["intercept"] + sum(
                (f - mu) / sd * c for f, mu, sd, c in
                zip(feats, memb["scaler_mean"], memb["scaler_scale"], memb["coef"]))
            zl += wt * z
        if self.calibration:
            a = self.calibration.get("platt_coef", 1.0)
            b = self.calibration.get("platt_intercept", 0.0)
            zl = a * zl + b
        return 1.0 / (1.0 + math.exp(-zl))

    def classify(self, text: str) -> dict:
        raw = [fn(text) for fn in FEATURE_FNS.values()]
        n_words = len(text.split())
        prob = self.probability(raw)
        return {
            "is_ai": prob >= self.threshold,
            "confidence": round(prob, 4),
            "answer": 1 if prob >= self.threshold else 0,
            "model": self.version,
            "reliable": n_words >= 50,
            "note": None if n_words >= 50 else (
                f"short text ({n_words} words): features unreliable below ~50 words; "
                "treat confidence as low-evidence"),
            "signals": dict(zip(FEATURE_NAMES, (round(v, 4) for v in raw))),
        }


_default = None


def get_classifier() -> SlopClassifier:
    global _default
    if _default is None:
        _default = SlopClassifier()
    return _default


def classify(text: str) -> dict:
    return get_classifier().classify(text)
