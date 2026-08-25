"""Episode runner + ablation driver for the slop-reduction lab.

Stages:
  A: payload variants {none, counts, locate, highlight, full} x 4 shared essays,
     full-rewrite mode. Paired design: every arm rewrites the same drafts.
  B: rewrite modes {full_rewrite, patch} at the winning payload.

All results append to experiments/lab/results.jsonl (resumable by key).

Run stages in background:
  nohup python3 lab/ablation.py --stage A > experiments/lab/A.log 2>&1 &
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "miner"))

from miner.src.classifier import classify
from miner.src.detector import detect
from lab.payloads import (VARIANTS, build_full_rewrite_prompt, build_patch_prompt,
                          matches_for)

OUT = os.path.join(ROOT, "experiments", "lab")
RESULTS = os.path.join(OUT, "results.jsonl")
MODEL_VERSION = json.load(open(os.path.join(ROOT, "miner/src/noslop_logreg_model.json"))).get("version")

TOPICS = ["love", "time", "memory", "justice"]
DRAFT_WORDS = 400


def hermes(prompt: str, timeout: int = 600) -> str:
    r = subprocess.run(["hermes", "-z", prompt], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"hermes failed: {r.stderr[-300:]}")
    return r.stdout.strip()


def analyze(text: str) -> dict:
    cls = classify(text)
    det = detect(text)
    n_sent = max(1, cls["signals"]["n_sent"])
    tags = {}
    for m in det.details:
        tags[m["tag"]] = tags.get(m["tag"], 0) + 1
    return {"answer": cls["answer"], "confidence": cls["confidence"],
            "pattern_count": len(det.details),
            "pattern_rate": round(len(det.details) / n_sent, 4),
            "tags": tags, "n_words": len(text.split())}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.!?\n]+[.!?]*", text) if s.strip()]


def flagged_sentences(text: str, cap: int = 20) -> list[dict]:
    sents = split_sentences(text)
    out, used = [], set()
    for m in matches_for(text):
        for i, s in enumerate(sents):
            if m["text"] in s or (m["text"].strip() and m["text"][:40] in s):
                key = (i, m["tag"])
                if key not in used:
                    used.add(key)
                    out.append({"id": f"s{i}", "sentence": s, "tag": m["tag"]})
                break
    # dedupe by sentence id (one fix per sentence, first tag wins)
    seen, dedup = set(), []
    for f in out:
        if f["id"] not in seen:
            seen.add(f["id"])
            dedup.append(f)
    return dedup[:cap]


def parse_json_loose(s: str) -> dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON found")
    return json.loads(s[start:end + 1])


def splice(text: str, replacements: dict) -> tuple[str, int]:
    applied = 0
    for sid, rep in replacements.items():
        idx = sid[1:]
        try:
            sents = split_sentences(text)
            old = sents[int(idx)]
        except (ValueError, IndexError):
            continue
        if old in text and isinstance(rep, str) and 10 < len(rep) < len(old) * 4 + 80:
            text = text.replace(old, rep.strip(), 1)
            applied += 1
    return text, applied


def episode(key: str, cfg: dict, v0: str) -> dict:
    """Run one config on one draft; up to MAX_ITERS guided rewrites."""
    MAX_ITERS = 2
    ep_dir = os.path.join(OUT, "episodes")
    os.makedirs(ep_dir, exist_ok=True)
    safe = key.replace("|", "_")
    recs = [dict(iteration=0, **analyze(v0))]
    cur = v0
    open(os.path.join(ep_dir, f"{safe}_v0.txt"), "w").write(cur)
    calls = 1  # the draft itself
    t0 = time.time()
    for it in range(1, MAX_ITERS + 1):
        prev = recs[-1]
        if prev["pattern_count"] == 0:
            break
        if cfg["mode"] == "full_rewrite":
            prompt = build_full_rewrite_prompt(cfg["payload"], cur)
            new = hermes(prompt)
            if len(new.split()) < 30:
                break
            calls += 1
        else:
            flagged = flagged_sentences(cur)
            if not flagged:
                break
            pj = build_patch_prompt(cfg["payload"], flagged)
            try:
                reps = parse_json_loose(hermes(pj))
            except (ValueError, json.JSONDecodeError):
                break
            cur2, applied = splice(cur, reps)
            calls += 1
            if applied == 0:
                break
            new = cur2
        an = analyze(new)
        recs.append(dict(iteration=it, **an))
        open(os.path.join(ep_dir, f"{safe}_v{it}.txt"), "w").write(new)
        cur = new

    first, last = recs[0], recs[-1]
    result = {
        "key": key,
        "config": cfg,
        "versions": recs,
        "hermes_calls_total": calls,
        "pattern_reduction_pct": round(
            100 * (first["pattern_count"] - last["pattern_count"])
            / max(1, first["pattern_count"]), 1),
        "rate_reduction_pct": round(
            100 * (first["pattern_rate"] - last["pattern_rate"])
            / max(first["pattern_rate"], 1e-9), 1),
        "length_retention_pct": round(100 * last["n_words"] / max(1, first["n_words"]), 1),
        "clean": last["pattern_count"] == 0,
        "final_confidence": last["confidence"],
        "wall_s": round(time.time() - t0, 1),
        "model_version": MODEL_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result, cur


def load_done() -> set[str]:
    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    return done


def append(res: dict):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["draft", "A", "B", "large"], required=True)
    ap.add_argument("--arm-limit", type=int, default=0, help="limit arms (debug)")
    ap.add_argument("--only", default="", help="comma-separated payload filter")
    ap.add_argument("--payload", default="highlight")
    ap.add_argument("--mode", default="patch")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    done = load_done()

    if args.stage == "draft":
        rng = __import__("random").Random(7)
        drafts = {}
        for topic in TOPICS:
            path = os.path.join(OUT, f"draft_{topic}.txt")
            if os.path.exists(path):
                drafts[topic] = open(path).read()
                continue
            print(f"drafting {topic}...", flush=True)
            p = (f"Write a {DRAFT_WORDS}-word reflective essay on {topic}. "
                 f"Write it well, as a thoughtful essayist. Output ONLY the essay.")
            t = hermes(p)
            open(path, "w").write(t)
            drafts[topic] = t
        print("drafts ready:", {k: len(v.split()) for k, v in drafts.items()})
        return

    if args.stage == "A":
        variants = VARIANTS[:args.arm_limit] if args.arm_limit else VARIANTS
        if args.only:
            variants = [v for v in variants if v in args.only.split(",")]
        for variant in variants:
            for topic in TOPICS:
                key = f"A|{variant}|{topic}"
                if key in done:
                    print(f"skip {key}")
                    continue
                v0 = open(os.path.join(OUT, f"draft_{topic}.txt")).read()
                print(f"episode {key}...", flush=True)
                res, _ = episode(key, {"stage": "A", "payload": variant,
                                       "mode": "full_rewrite"}, v0)
                append(res)
                print(f"  -> reduction={res['pattern_reduction_pct']}% "
                      f"retention={res['length_retention_pct']}% clean={res['clean']}",
                      flush=True)
        return

    if args.stage == "B":
        best_payload = args.payload
        for mode in ("full_rewrite", "patch"):
            for topic in TOPICS:
                key = f"B|{best_payload}|{mode}|{topic}"
                if key in done:
                    print(f"skip {key}")
                    continue
                v0 = open(os.path.join(OUT, f"draft_{topic}.txt")).read()
                print(f"episode {key}...", flush=True)
                res, _ = episode(key, {"stage": "B", "payload": best_payload,
                                       "mode": mode}, v0)
                append(res)
                print(f"  -> reduction={res['pattern_reduction_pct']}% "
                      f"retention={res['length_retention_pct']}% "
                      f"calls={res['hermes_calls_total']}", flush=True)
        return

    if args.stage == "large":
        src = "/root/projects/workengestation/output/essays/tantra/tantraloka-essays/daimon-contact.md"
        body = "\n".join(l for l in open(src).read().split("\n")
                         if not l.strip().startswith("#"))
        v0 = " ".join(body.split())[:9000]
        payload = args.payload
        mode = args.mode
        key = f"large|{payload}|{mode}"
        if key in done:
            print("already done")
            return
        res, final = episode(key, {"stage": "large", "payload": payload,
                                   "mode": mode}, v0)
        res["source_file"] = src
        open(os.path.join(OUT, "large_final.txt"), "w").write(final)
        append(res)
        print(f"-> reduction={res['pattern_reduction_pct']}% "
              f"retention={res['length_retention_pct']}%", flush=True)


if __name__ == "__main__":
    main()
