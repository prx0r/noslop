# NoSlop — Repository Index

Every file and directory, labelled. Status codes:
`[ACTIVE]` production/training pipeline · `[DOC]` documentation ·
`[DATA]` datasets/results · `[LAB]` experiment infrastructure ·
`[ARCHIVE]` superseded, kept for provenance

## Entry points
| Path | Status | Purpose |
|---|---|---|
| `README.md` | DOC | project overview + quick start |
| `CANONICAL-PLAN-2026-08-25.md` | DOC | **canonical plan** — facts, roadmap, operating rules |
| `PROJECT-STATUS.md` | DOC | current state snapshot |
| `EXPERIMENTAL-REPORT.md` | DOC | full experimental report w/ file references |
| `PEER-REVIEW.md` | DOC | frontier peer review + revision log (R1–R4 executed) |
| `COMPETITOR-ANALYSIS.md` | DOC | 8-repo competitive analysis |
| `INDEX.md` | DOC | this file |

## Serving pipeline (`miner/src/`)
| Path | Status | Purpose |
|---|---|---|
| `miner/src/classifier.py` | ACTIVE | pure-python ensemble inference + Platt calibration + short-text flag |
| `miner/src/noslop_logreg_model.json` | ACTIVE | frozen serving artifact: members + manifest sha256 + serving_params + calibration |
| `miner/src/features.py` | ACTIVE | 41 canonical signal functions (shared train/serve) |
| `miner/src/detector.py` | ACTIVE | v6 pattern engine (NARR/NEG/3LIST/CLICHE/FLAT), span-deduped |
| `miner/src/knowledge.py` | ACTIVE | pattern KB: definitions/disguises/repairs/budgets; `glossary()` + `glossary_compact()` |
| `miner/src/server.py` | ACTIVE | free HTTP endpoint |
| `miner/src/x402_server.py` | ACTIVE | x402-gated endpoint ($0.003/detect) |
| `miner/src/test_detector.py` | ACTIVE | pattern-engine tests |
| `miner/src/x402_client_test.py` | ACTIVE | payment-flow test (ALL PASS) |

## Training & evaluation pipeline (repo root)
| Path | Status | Purpose |
|---|---|---|
| `train_logreg.py` | ACTIVE | trains both ensemble members; writes artifact to `miner/src/`; replace-if-wins comparisons |
| `cogym_optimize.py` | ACTIVE | cogym-kernel tuning of ensemble weight + threshold; writes `serving_params` into artifact |
| `calibrate.py` | ACTIVE | Platt scaling + human-pool FPR promise (~1%) → artifact `calibration` block |
| `eval_metrics.py` | ACTIVE | five-metric suite (AUROC/AUPRC/Brier/EER-acc/TPR@FPR1%), length buckets, permutation importance |
| `bench_run.py` | ACTIVE | full benchmark suite across all datasets |
| `evolve_heldout.py` | ACTIVE* | shared utils (wilson, search space, sample/mutate) — name is historical; imported by active scripts |

## Experiment infrastructure
| Path | Status | Purpose |
|---|---|---|
| `lab/payloads.py` | LAB | guidance-payload ladder builders (none/counts/locate/highlight/full) |
| `lab/ablation.py` | LAB | episode runner: stages draft/A/B/large; resumable JSONL |
| `lab/analyze.py` | LAB | Wilson CIs + paired win/tie/loss analysis |
| `slop_loop.py` | LAB | single-topic write→diagnose→rewrite assessor |
| `agent_demo.py` | LAB | real workstation texts through the paid x402 flow |
| `experiments/lab/results.jsonl` | LAB | episode ledger (append-only, resumable by key) |
| `experiments/lab/draft_*.txt` | DATA | shared baseline essays for paired ablations |
| `experiments/lab/episodes/` | DATA | per-episode rewritten texts |
| `experiments/LABS.md` | DOC | running experiment log with interim findings |

## Data
| Path | Status | Purpose |
|---|---|---|
| `data/test_dataset.json` | DATA | original 55-entry set (in-distribution reference) |
| `data/make_split.py` | ACTIVE | seed-frozen stratified splitter |
| `data/train_split.json`, `data/heldout_split.json` | DATA | first-generation splits (superseded by benchmark-era splits, kept) |
| `data/benchmarks/hc3_bench.json` | DATA | 5,575 HC3 human/ChatGPT |
| `data/benchmarks/semeval_bench.json` | DATA | 3,600 SemEval-2024 T8 (OOD generators) |
| `data/benchmarks/polished_ai_bench.json` | DATA | 39 v7-edited AI essays (adversarial) |

## Results snapshots (`experiments/results/`)
| File | Content |
|---|---|
| `noslop_benchmark_results.json` | full benchmark suite numbers |
| `noslop_cogym_optimize_results.json` | champion config + untouched-test metrics |
| `noslop_frontier_metrics.json` | five-metric suite, length buckets, NEG sign audit |
| `noslop_logreg_model_training.json` | training-time model snapshot |
| `noslop_cv_results.json`, `noslop_heldout_results.json`, `noslop_binary_evolution_results.json` | historical GA/CV lineage results |

## Reference corpus (inputs to knowledge base)
| Path | Status |
|---|---|
| `algorithms/{v6,v7,longform}-algorithm.md` | detection/repair doctrine (zero-tolerance vs purpose-tested regimes) |
| `anti-slop/{stopslop,antislop,antislopguide}.md` | repair toolbox + never-rules |
| `quality/goodprose.md`, `quality/rumiengine.md` | positive prose targets, Rumi mechanics |
| `detection/slopnotes.md`, `detection/slopreview.md` | audit notes, extended taxonomy |
| `audit/workshopalgorithm.md`, `audit/essaygenaudit.md` | workshop protocol, meta-map |
| `worldpacks/{noslop,weather}/` | cogym worldpack wrappers (noslop used by evolve_noslop_v2 lineage) |
| `repos/cogym/` | vendored cogym kernel (imported by cogym_optimize/ablation lineage) |

## Archive (`archive/legacy-ga/`)
Superseded genetic-algorithm lineage and pre-benchmark weather experiments,
retained for provenance: `evolve_noslop.py`, `evolve_noslop_v2.py`,
`evolve_weather.py`, `cv_heldout.py`, `evolve_binary.py`,
`miner/weather_server.py`, first-generation result JSONs.
