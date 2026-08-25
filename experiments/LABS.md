# NoSlop Labs — Experiment Log

All runs append to `experiments/lab/results.jsonl` (one JSON per episode,
keyed `stage|payload[|mode]|topic`). Resumable: completed keys are skipped.

## Environment

| Field | Value |
|---|---|
| Writer/rewriter | `hermes -z` (OpenCode Go; ox-alpha-free quota exhausted -> mimo-v2.5 fallback) |
| Miner | logreg_v3 ensemble 28 signals + v6 pattern engine |
| Drafts | 4 shared essays (love/time/memory/justice, ~400 words each), drafted once, reused across arms — paired design |
| Baseline drafts | love: 18 patterns (13 NEG), time: 15 (11 NEG), memory: 16 (13 NEG), justice: 17 (11 NEG); classifier conf < 0.02 |

## Payload information ladder

```
none     "remove AI slop" generic instruction only
counts   + aggregate tag counts
locate   + flagged passages with tags
highlight+ why each pattern reads as machine-generated
full     + suggested repairs + long-form budgets + sequencing rules
```

## Stage A — payload ablation (full-rewrite mode)

Status: COMPLETE (18/20 episodes; counts|justice + one other straggler).

| arm | n | red% | ret% | clean | paired W/T/L vs none |
|---|---|---|---|---|---|
| none | 4 | 35.9 | 87.4 | 0/4 | — |
| counts | 2+ | 82.2 | 90.8 | 0/2 | 2/0/0 |
| locate | 4 | 94.0 | 94.2 | 1/4 | 4/0/0 |
| **highlight** | **4** | **97.0** | 92.5 | **2/4** | **4/0/0** |
| full | 4 | 84.3 | **73.8** | 1/4 | 3/0/1 |

FINDINGS (Stage A):
1. Every diagnostic arm beats the generic control — diagnosis is the product.
2. `highlight` wins: what+where+why, no scripted repairs, no rule dumps.
3. `full` is DOMINATED on both axes: less reduction AND worse retention
   (one episode gutted the text to 73.8%). The heavy JSON payload with
   repair scripts/budgets/rules actively hurts a capable rewriter.
   -> Supports the minimum-payload philosophy; serving default should move
   to highlight-shaped guidance pending stage B confirmation.
4. Payload economics (draft_love, 18 matches): counts=22 tok, locate=180,
   highlight=236, full=1639. The winning payload costs ~1/7 of the losing
   one — less guidance, less latency, better outcomes.

## Stage B — rewrite mode at payload=highlight

Status: RUNNING. Interim:

| mode | n | red% | ret% | clean | calls/ep |
|---|---|---|---|---|---|
| full_rewrite | 4 | 92.0 | 89.9 | 1/4 | 2.0 |
| patch | 0 | — | — | — | — |

Patch = large-doc path: cost scales with flagged sentences, untouched prose
preserved exactly by construction. Risk under test: does hermes return clean
JSON replacements, and do spliced rewrites read coherently?

Hypotheses:
- H1: highlight >= full on reduction (repairs/rules add little for a capable LLM)
- H2: all diagnostic arms > none
- H3: length retention roughly equal across arms (~90%+) — slop removal
  should not gut content

## Stage B — rewrite mode at best payload

Pending A. full_rewrite vs patch (sentence-level splice via single JSON call).
Patch is the large-doc path: O(flagged sentences) tokens instead of
O(document); also preserves untouched prose exactly.

## Large-doc validation

daimon-contact.md (~660 words, v7-polished AI) at winning config.

## Integration lab — competitor feature imports (2026-08-25)

Cloned and inspected 8 competitor repos (`/root/competitors/`, analysis in
`COMPETITOR-ANALYSIS.md`). Ported 13 signals from slop-cop + dslop into
`miner/src/features.py`: staccato-burst, anaphora-abuse, negation-countdown,
colon-elaboration, question-then-answer, superficial-analysis frames,
sentence-start anchored openers, vague-attribution, parenthetical-qualifier,
almost-hedge, listicle magic-number (3/5/7/10), sentence-length kurtosis,
word-frequency concentration. 41 signals total, +6ms per call.

Replace-if-wins gate (slopscore's promotion rule): v4 must hold or improve
held-out HC3 AND SemEval OOD.

| set | v3 (28 sig) | v4 (41 sig) | verdict |
|---|---|---|---|
| heldout HC3 | 86.8% [84.2–88.9] | 87.0% [84.5–89.2] | holds |
| original set | 81.8% | 85.5% | +3.7 |
| SemEval OOD | 84.0% | **86.5%** [85.2–87.8] | **+2.5** |

PROMOTED. `listicle3_5_7_10` emerged as a top-4 blended coefficient (+0.96) —
the magic-number-list tell carries real classification weight.

cogym re-optimization on v4: champion w_specialist=0.25 threshold=0.58
(val_macro 0.8775). Untouched test: HC3 87.2% [84.8–89.4], SemEval 86.1%
[84.5–87.5]. Full HC3 run: 88.3% [87.4–89.1] (n=5575). x402 flows re-verified
ALL PASS after redeploy.

## Notes / deviations

- mimo-v2.5 fallback makes hermes calls 1-4 min; arms run as parallel
  processes to compensate. Episode results append incrementally, so partial
  analysis is valid at any time.
