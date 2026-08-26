# NoSlop — Project Status

Updated: 2026-08-25 (session: miner hardening -> x402 -> cogym optimization -> slop-reduction lab)

## What NoSlop is

Two-layer AI-text analysis service, built for the Telegraph Protocol Hackathon
(AI_TEXT_DETECTION intent, deadline Sep 7 2026) and for a Track-3 "unslop agent"
application:

1. **Classifier** — `logreg_v3` ensemble (2 logistic members, 28 stylometric +
   v6-pattern signals), pure-python inference. Optimized by cogym kernel
   (`elitist_mutation`) to serving params w=0.0 specialist / threshold=0.48.
2. **Diagnosis layer** — v6 regex pattern engine (NARR/NEG/3LIST/CLICHE/FLAT)
   with span-dedup + `knowledge.py`: per-pattern definitions, disguise catalogs,
   ranked repairs, regime budgets distilled from the writing corpus
   (v6/v7 algorithms, stopslop/goodprose/rumiengine).

## Honest performance

| Benchmark | n | Accuracy | Note |
|---|---|---|---|
| HC3 held-out | 800 | 83.8% [81.0–86.1] | cogym-chosen config; never trained/tuned on |
| SemEval-2024 T8 unseen slice | 2000 | 86.9% [85.4–88.3] | OOD generators (chatGPT/cohere/davinci) |
| Original slop set | 55 | ~78–80% | in-distribution reference |
| Polished v7 essays | 39 | ~0–8% detected | engineered-indistinguishable text — expected |

Single-domain training collapsed OOD to 54%; multi-domain ensemble fixed it.
Lesson recorded: domain breadth beats home-turf accuracy.

## Deployment

```
miner/src/
├── classifier.py            ensemble inference (no numpy/sklearn needed)
├── noslop_logreg_model.json frozen artifact + manifest sha256 + serving_params
├── detector.py              pattern engine (deduped spans)
├── knowledge.py             agent-facing fix guidance (5 payload variants in lab/)
├── features.py              28 canonical signal functions
├── server.py                free HTTP endpoint
└── x402_server.py           paid endpoint: /.well-known/x402 discovery,
                             402 challenge, verify/settle (mock|live facilitator),
                             $0.003/detect Base Sepolia USDC
```

Test suites: `x402_client_test.py` (full payment flow, ALL PASS),
`agent_demo.py` (real workstation texts through paid flow),
`slop_loop.py` (write->diagnose->rewrite loop).

## Lab infrastructure

```
lab/
├── payloads.py     information ladder: none < counts < locate < highlight < full
├── ablation.py     episode runner + stages A/B/large (resumable JSONL)
└── analyze.py      Wilson CIs, paired win/tie/loss vs control
experiments/lab/    results.jsonl + drafts + logs    (LABS.md = experiment log)
cogym_optimize.py   serving-param optimization via cogym_kernel recipes+gates
evolve_binary.py    GA over legacy linear scorer (superseded by train_logreg)
train_logreg.py     trains both ensemble members; writes artifact + manifest
bench_run.py        benchmark suite (HC3/SemEval/polished/original)
data/benchmarks/    hc3_bench.json (5575), semeval_bench.json (3600), polished_ai_bench.json (39)
```

## Key decisions recorded

- Binary verdict is calibrated but the *product* is the diagnosis payload:
  agents pay for "what to fix", not for a coin-flip label on edited text.
- **Stage A lab result (n=20 episodes, paired)**: `highlight` payload
  (what+where+why, 236 tokens) beats `full` (repairs+budgets+rules, 1639
  tokens) — 97.0% vs 84.3% pattern reduction AND better length retention.
  Over-guiding measurably hurts capable rewriters. Serving default moves to
  compact guidance pending stage B (rewrite-mode) confirmation.
- Patch-mode rewriting (sentence splice via one JSON call) is the scalable
  path for large docs — cost scales with flagged sentences, not doc length.
- cogym integration: manifest hashing + serving-param optimization done;
  receipts/claims + secret-holdout recommended next (machinery exists).

## Open items

- Track 1 registration YAML (real wallet, live facilitator check)
- Stage A completion (~20 episodes), then B and large-doc validation
- Optional: wire bench runs through AsyncRunner -> CapabilityClaims

## Repository organization (2026-08-25 tidy)

Full labelled inventory: `INDEX.md`. Canonical roadmap:
`CANONICAL-PLAN-2026-08-25.md`. Legacy GA lineage + pre-benchmark results in
`archive/legacy-ga/`. Result snapshots live in `experiments/results/` (active
scripts write there). Root scripts are the active train→gate→optimize→deploy
chain and cross-import `evolve_heldout.py` utilities.

## Cross-lane state (2026-08-26)

Track 1 miner: submission-ready, blocked only on registry YAML + wallet YAML.
Track 2 fleet lane (separate repo /root/telegraph-lab): NUM family binaries
built/verified/beaten-champions-proven; submission runbook handed to wallet
lane; autoloop daemon live for verdict-driven evolution.
Track 3 unslop agent: skeleton ready (slop_loop.py), opens post T1/T2.
