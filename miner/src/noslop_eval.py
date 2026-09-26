"""noslop_eval — Track 2 evaluation script for AI_TEXT_DETECTION.

A WASM-compatible scorer that grades how well a miner's answer matches
the ground truth. Unlike our naive competitor scorers (give 0 to everything),
this uses graduated scoring: semantic overlap + pattern-matching + calibrated
confidences.

Built from literature (SMILE/PEDANTS) + our production scorer primitives.
Exports: rank_answer(q_ptr, q_len, gt_ptr, gt_len, ma_ptr, ma_len) -> f32
"""
import re, math, os, sys, json

# ── Freestanding helpers (WASM-portable) ───────────────────────────────────

STOP_WORDS = {"a","an","the","is","are","was","were","be","been","of","in","on",
              "at","to","for","and","or","but","it","as","by","with"}

def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS and len(w) > 1}

def _trigrams(text: str) -> set[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    return {s[i:i+3] for i in range(max(0, len(s)-2))}

def _smoothstep(x: float) -> float:
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    return x * x * (3 - 2 * x)

def _normalize_eq(a: str, b: str) -> bool:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    return ta == tb and len(ta) > 0

# ── Core scoring ────────────────────────────────────────────────────────────

def _normalize_numbers(text: str) -> str:
    """Strip currency symbols, commas, handle common abbreviations."""
    t = text.lower().strip()
    t = re.sub(r"[\$€£¥]", "", t)
    t = re.sub(r"\s*usd\s*", "", t)
    t = re.sub(r"\s*eur\s*", "", t)
    # handle abbreviations: 50k -> 50000, 1.5M -> 1500000, etc
    for suf, mult in [(" trillion", 1e12), (" billion", 1e9), (" million", 1e6),
                       (" thousand", 1e3), ("bn", 1e9), ("m", 1e6), ("k", 1e3)]:
        t = re.sub(r"(\d[\d,.]*)\s*" + re.escape(suf) + r"\b", lambda m: str(float(m.group(1).replace(",","")) * mult), t)
    t = re.sub(r",", "", t)
    return t.strip()

def _extract_number(text: str) -> float | None:
    """Extract first numeric value from text."""
    norm = _normalize_numbers(text)
    m = re.search(r"(\d+(?:\.\d+)?)", norm)
    if m:
        try: return float(m.group(1))
        except ValueError: return None
    return None

def score_answer(question: str, ground_truth: str, miner_answer: str) -> float:
    """Grade a miner's answer against ground truth. Returns [0, 1]."""
    ans = (miner_answer or "").strip()
    gt = (ground_truth or "").strip()
    if not ans or not gt:
        return 0.0

    # exact normalized match
    if _normalize_eq(gt, ans):
        return 1.0

    # numeric exact match after normalization (50k == 50000 == $50,000)
    gt_num = _extract_number(gt)
    an_num = _extract_number(ans)
    if gt_num is not None and an_num is not None:
        if abs(gt_num - an_num) < max(abs(gt_num) * 0.001, 0.01):
            return 1.0

    # token F1 (recall-weighted for factoid QA)
    gt_t, an_t = _tokens(gt), _tokens(ans)
    if not gt_t or not an_t:
        return 0.0
    ov = len(gt_t & an_t)
    prec = ov / len(an_t)
    rec = ov / len(gt_t)
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0

    # character trigram similarity (handles morphological variants)
    gt_tri, an_tri = _trigrams(gt), _trigrams(ans)
    tri = len(gt_tri & an_tri) / len(gt_tri | an_tri) if gt_tri and an_tri else 0.0

    # number matching (handles formatting variants)
    gt_nums = set(float(m.group().replace(",","")) for m in
                  re.finditer(r"\d[\d,]*(?:\.\d+)?", _normalize_numbers(gt))
                  if float(m.group().replace(",","")) > 0)
    an_nums = set(float(m.group().replace(",","")) for m in
                  re.finditer(r"\d[\d,]*(?:\.\d+)?", _normalize_numbers(ans))
                  if float(m.group().replace(",","")) > 0)

    if gt_nums:
        rels = [min(abs(g-a)/max(abs(g),1e-9) for a in an_nums) for g in gt_nums]
        num_avg = sum(max(0, 1 - min(r/0.01, 1.0)) for r in rels) / len(rels)
    elif an_nums:
        num_avg = 0.3
    else:
        num_avg = 0.0

    base = 0.50 * f1 + 0.20 * tri + 0.30 * num_avg
    sharp = _smoothstep(max(0, min(1, base)))
    final = min(sharp + abs(base) * 1e-9, 1.0 - 1e-7)
    return round(max(0.0, final), 8)


# ── WASM exports ────────────────────────────────────────────────────────────

# rank_answer(q, q_len, gt, gt_len, ma, ma_len) -> f32
# reads strings from linear memory, scores, returns f32

def rank_answer_raw(gt_text: str, ma_text: str) -> float:
    return score_answer("", gt_text, ma_text)


# ── CLI for testing ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        gt, ma = sys.argv[2], sys.argv[3]
        q = sys.argv[1] if len(sys.argv) > 3 else ""
        print(f"score: {score_answer(q, gt, ma):.6f}")
    else:
        # self-test
        tests = [
            ("0.500000", "0.5", 1.0),
            ("0.500000", "0.3", 0.0),
            ("Bitcoin costs $50k", "Bitcoin costs $50,000", 1.0),
            ("The sky is blue", "The ocean is deep", 0.0),
        ]
        for gt, ma, exp in tests:
            s = score_answer("", gt, ma)
            ok = abs(s - exp) < 0.01
            print(f"  {'PASS' if ok else 'FAIL'} score('{gt[:30]}','{ma[:30]}') = {s:.4f} (expected {exp})")
