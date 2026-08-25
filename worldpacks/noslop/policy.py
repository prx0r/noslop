"""Classifier policy: uses config weights to classify text as AI or human.

Config is the gene being evolved:
- pattern weights (narr, neg, three_list, cliche)
- statistical thresholds (pattern_threshold, burstiness_threshold, punct_threshold)
"""
from __future__ import annotations

from cogym_kernel.kernel.contracts import ActionSpec, PolicyDecision

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from miner.src.features import extract


# Search space for evolution
SEARCH_SPACE = {
    "narr_weight": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "neg_weight": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "three_list_weight": [0.3, 0.5, 0.8, 1.0, 1.2],
    "cliche_weight": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    "pattern_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    "burstiness_threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
    "punct_threshold": [0.01, 0.02, 0.03, 0.04, 0.05],
    "ai_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
}


class ClassifierPolicy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.policy_id = f"classifier_{hash(str(sorted(self.config.items())))}"

    def initialize(self, world_spec):
        return {}

    def act(self, obs: dict, actions: tuple[ActionSpec, ...], pstate: dict):
        if not actions:
            return None

        action = actions[0]
        text = action.payload["text"]

        weights = {
            "narr": self.config.get("narr_weight", 1.0),
            "neg": self.config.get("neg_weight", 1.0),
            "three_list": self.config.get("three_list_weight", 1.0),
            "cliche": self.config.get("cliche_weight", 1.0),
            "pattern_threshold": self.config.get("pattern_threshold", 0.5),
            "burstiness_threshold": self.config.get("burstiness_threshold", 0.3),
            "punct_threshold": self.config.get("punct_threshold", 0.02),
            "ld_threshold": self.config.get("ld_threshold", 0.3),
            "ngram_threshold": self.config.get("ngram_threshold", 0.5),
            "fw_threshold": self.config.get("fw_threshold", 0.45),
            "the_threshold": self.config.get("the_threshold", 0.35),
            "abstract_threshold": self.config.get("abstract_threshold", 0.6),
            "commitment_threshold": self.config.get("commitment_threshold", 0.1),
        }

        fv = extract(text, weights)
        threshold = self.config.get("ai_threshold", 0.5)
        prediction = 1 if fv.confidence >= threshold else 0

        return PolicyDecision(
            action=ActionSpec(
                kind="CLASSIFY",
                payload={
                    "text": text,
                    "prediction": prediction,
                    "confidence": fv.confidence,
                    "features": {
                        "narr": fv.narr_count,
                        "neg": fv.neg_count,
                        "three_list": fv.three_list_count,
                        "cliche": fv.cliche_count,
                        "total_patterns": fv.total_patterns,
                        "pattern_rate": round(fv.pattern_rate, 4),
                        "burstiness": round(fv.burstiness, 4),
                        "vocab_diversity": round(fv.vocab_diversity, 4),
                        "punct_density": round(fv.punctuation_density, 4),
                        "fragment_ratio": round(fv.fragment_ratio, 4),
                    },
                },
                executor_kind="deterministic",
            ),
            rationale=f"conf={fv.confidence:.3f} pred={prediction}",
        )
