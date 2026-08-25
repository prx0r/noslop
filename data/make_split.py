"""Create stratified train/held-out split of the noslop test dataset.

Seed-frozen so the split itself is reproducible and tamper-evident.
Run: python3 data/make_split.py
"""
import hashlib
import json
import os
import random

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "test_dataset.json")
TRAIN_OUT = os.path.join(HERE, "train_split.json")
HELDOUT_OUT = os.path.join(HERE, "heldout_split.json")

SEED = 42
HELDOUT_FRACTION = 0.40


def main():
    with open(SRC) as f:
        dataset = json.load(f)

    entries = dataset["entries"]
    by_label = {0: [], 1: []}
    for e in entries:
        by_label[e["label"]].append(e)

    rng = random.Random(SEED)
    train, heldout = [], []
    for label in (0, 1):
        group = sorted(by_label[label], key=lambda e: e["id"])
        rng.shuffle(group)
        n_heldout = round(len(group) * HELDOUT_FRACTION)
        heldout.extend(group[:n_heldout])
        train.extend(group[n_heldout:])

    train.sort(key=lambda e: e["id"])
    heldout.sort(key=lambda e: e["id"])

    manifest = {
        "source": "test_dataset.json",
        "seed": SEED,
        "heldout_fraction": HELDOUT_FRACTION,
        "train_size": len(train),
        "heldout_size": len(heldout),
        "train_ids": [e["id"] for e in train],
        "heldout_ids": [e["id"] for e in heldout],
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()[:16]

    with open(TRAIN_OUT, "w") as f:
        json.dump({"entries": train, **manifest}, f, indent=2)
    with open(HELDOUT_OUT, "w") as f:
        json.dump({"entries": heldout, **manifest}, f, indent=2)

    print(f"train:   {len(train)} ({sum(e['label'] for e in train)} ai)")
    print(f"heldout: {len(heldout)} ({sum(e['label'] for e in heldout)} ai)")
    print(f"manifest_hash: {manifest['manifest_hash']}")


if __name__ == "__main__":
    main()
