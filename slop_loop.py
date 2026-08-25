"""Slop-reduction loop experiment: hermes writes -> miner diagnoses -> hermes
rewrites from diagnosis -> re-measure.

Measures how useful the miner's guidance payload is to a real agent by the
only metric that matters: does following it reduce measured slop without
gutting the text?

Outputs experiments/<slug>/ with every version + metrics + manifest.

Run: python3 slop_loop.py --topic "love" --words 400 --iters 2
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "miner"))
from miner.src.classifier import classify
from miner.src.detector import detect
from miner.src.knowledge import glossary

HERE = os.path.dirname(os.path.abspath(__file__))


def hermes(prompt: str) -> str:
    t0 = time.perf_counter()
    r = subprocess.run(["hermes", "-z", prompt], capture_output=True, text=True,
                       timeout=600)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"hermes failed: {r.stderr[-500:]}")
    return r.stdout.strip(), dt


def analyze(text: str) -> dict:
    cls = classify(text)
    det = detect(text)
    seen = sorted({m["tag"] for m in det.details})
    n_sent = max(1, cls["signals"]["n_sent"])
    return {
        "answer": cls["answer"],
        "confidence": cls["confidence"],
        "pattern_count": len(det.details),
        "pattern_rate": round(len(det.details) / n_sent, 4),
        "tags": {t: sum(1 for m in det.details if m["tag"] == t) for t in seen},
        "n_words": len(text.split()),
        "n_sentences": n_sent,
    }


def build_rewrite_prompt(text: str, analysis: dict) -> str:
    g = glossary(sorted(analysis["tags"]), regime="long")
    det = detect(text)
    lines = []
    for m in det.details[:25]:
        entry = {"tag": m["tag"], "matched_text": m["text"]}
        k = glossary([m["tag"]], "long")["patterns"].get(m["tag"], {})
        if k:
            entry["why_slop"] = k["why_slop"]
            entry["repairs"] = [r["hint"] for r in k["repairs"][:2]]
        lines.append(entry)

    rules = "\n".join(f"- {r}" for r in g["sequencing_rules"])
    return f"""You are revising an essay to remove AI-slop writing patterns while preserving its meaning, voice, and length.

DIAGNOSTIC REPORT (from a pattern detector):
{json.dumps(lines, indent=1)}

PATTERN BUDGETS (long-form):
{json.dumps({k: v for k, v in g['patterns'].items()}, indent=1)[:2200]}

REPAIR SEQUENCING RULES:
{rules}

ORIGINAL ESSAY:
{text}

TASK: Rewrite the essay fixing every flagged pattern using the suggested repair strategies. Keep the same argument, imagery, and roughly the same length ({analysis['n_words']} words +/- 10%). Do not introduce new listed patterns. Prefer concrete nouns over abstractions. Output ONLY the rewritten essay - no preamble, no commentary, no explanations."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="love")
    ap.add_argument("--words", type=int, default=400)
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()

    slug = args.topic.lower().replace(" ", "-")
    outdir = os.path.join(HERE, "experiments", f"slop-loop-{slug}")
    os.makedirs(outdir, exist_ok=True)

    print(f"=== SLOP LOOP: topic='{args.topic}' iters={args.iters} ===")

    write_prompt = (f"Write a {args.words}-word reflective essay on {args.topic}. "
                    f"Write it well, as a thoughtful essayist. Output ONLY the essay text "
                    f"- no title, no preamble, no commentary.")
    print("\n[1] hermes drafting...")
    v0, dt = hermes(write_prompt)
    print(f"    drafted {len(v0.split())} words in {dt:.0f}s")
    open(os.path.join(outdir, "v0.txt"), "w").write(v0)

    versions, records = [v0], []
    a = analyze(v0)
    rec = {"version": 0, "latency_s": round(dt, 1), **a}
    records.append(rec)
    print(f"    v0: conf={a['confidence']} patterns={a['pattern_count']} "
          f"rate={a['pattern_rate']} tags={a['tags']}")

    for i in range(1, args.iters + 1):
        prev = versions[-1]
        prev_rec = records[-1]
        if prev_rec["pattern_count"] == 0 and prev_rec["confidence"] < 0.5:
            print(f"\n[{i}] clean already - stopping loop")
            break
        print(f"\n[{i}] building diagnosis + hermes rewriting...")
        prompt = build_rewrite_prompt(prev, prev_rec)
        open(os.path.join(outdir, f"v{i-1}_rewrite_prompt.txt"), "w").write(prompt)
        vn, dt = hermes(prompt)
        vn = vn.strip()
        if len(vn.split()) < 40:
            print("    WARNING: suspiciously short rewrite, keeping previous")
            vn = prev
        an = analyze(vn)
        open(os.path.join(outdir, f"v{i}.txt"), "w").write(vn)
        versions.append(vn)
        rec = {"version": i, "latency_s": round(dt, 1), **an}
        records.append(rec)
        d_pat = an["pattern_count"] - prev_rec["pattern_count"]
        d_conf = round(an["confidence"] - prev_rec["confidence"], 4)
        print(f"    v{i}: conf={an['confidence']} ({d_conf:+}) patterns={an['pattern_count']} "
              f"({d_pat:+}) rate={an['pattern_rate']} tags={an['tags']} words={an['n_words']}")

    first, last = records[0], records[-1]
    summary = {
        "pattern_reduction_pct": round(
            100 * (first["pattern_count"] - last["pattern_count"])
            / max(1, first["pattern_count"]), 1),
        "pattern_rate_reduction_pct": round(
            100 * (first["pattern_rate"] - last["pattern_rate"])
            / max(first["pattern_rate"], 1e-9), 1),
        "confidence_delta": round(last["confidence"] - first["confidence"], 4),
        "length_retention_pct": round(100 * last["n_words"] / max(1, first["n_words"]), 1),
        "versions": records,
    }
    manifest = {
        "topic": args.topic,
        "target_words": args.words,
        "iterations": args.iters,
        "writer": "hermes -z",
        "miner": json.load(open("miner/src/noslop_logreg_model.json")).get("version"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifact_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()).hexdigest()[:16],
    }
    json.dump({"manifest": manifest, "summary": summary, "versions": records},
              open(os.path.join(outdir, "results.json"), "w"), indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"pattern reduction : {summary['pattern_reduction_pct']}%")
    print(f"rate reduction    : {summary['pattern_rate_reduction_pct']}%")
    print(f"confidence delta  : {summary['confidence_delta']:+}")
    print(f"length retention  : {summary['length_retention_pct']}%")
    print(f"saved -> {outdir}/results.json")


if __name__ == "__main__":
    main()
