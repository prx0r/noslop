"""NoSlop classifier worldpack: evolve optimal detection weights via cogym.

World: labeled text dataset (human vs AI)
Candidate: classifier config (pattern weights + thresholds)
Quality gate: accuracy ≥ 0.80
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from cogym_kernel.kernel.contracts import (
    ActionResult, ActionSpec, Metric, MetricVector, WorldSpec
)

# Import the feature extractor
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from miner.src.features import extract, FeatureVector


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_dataset.json")


@dataclass
class ClassifierState:
    text: str
    true_label: int
    features: FeatureVector | None = None
    prediction: int | None = None


class NoslopWorld:
    def __init__(self):
        self._spec = None
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            with open(DATASET_PATH) as f:
                self._dataset = json.load(f)["entries"]
        return self._dataset

    @property
    def world_spec(self) -> WorldSpec:
        if self._spec is None:
            self._spec = WorldSpec(
                world_kind="noslop.classifier",
                version="1",
                instance_set_hash="dataset-v1",
                environment_hash="features-v1",
                oracle_hash="labels-v1",
            )
        return self._spec

    @property
    def worldpack_id(self) -> str:
        from cogym_kernel.kernel.ids import content_id
        return content_id("wp", {"kind": "noslop.classifier", "v": 1})

    def reset(self, *, instance_id: str, seed: int) -> ClassifierState:
        dataset = self._load_dataset()
        entry = dataset[seed % len(dataset)]
        return ClassifierState(text=entry["text"], true_label=entry["label"])

    def observe(self, state: ClassifierState) -> dict:
        return {"text": state.text[:200] + "..." if len(state.text) > 200 else state.text}

    def actions(self, state: ClassifierState) -> tuple[ActionSpec, ...]:
        return (
            ActionSpec(
                kind="CLASSIFY",
                payload={"text": state.text},
                executor_kind="deterministic",
            ),
        )

    def apply(self, state: ClassifierState, action: ActionSpec,
              result: ActionResult) -> ClassifierState:
        if action.kind == "CLASSIFY" and result.status == "ok":
            return ClassifierState(
                text=state.text,
                true_label=state.true_label,
                features=result.payload.get("features"),
                prediction=result.payload.get("prediction"),
            )
        return state

    def terminal(self, state: ClassifierState) -> bool:
        return state.prediction is not None

    def score(self, state: ClassifierState) -> MetricVector:
        if state.prediction is None:
            return MetricVector(metrics=(
                Metric("accuracy", 0.0, "max"),
                Metric("correct", 0.0, "max"),
            ))
        correct = 1.0 if state.prediction == state.true_label else 0.0
        return MetricVector(metrics=(
            Metric("accuracy", correct, "max"),
            Metric("correct", correct, "max"),
        ))
