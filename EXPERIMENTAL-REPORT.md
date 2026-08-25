# NoSlop — Experimental Report

Date: 2026-08-25
Project root: `/root/noslop`
Status of record: `experiments/lab/results.jsonl` (28 episodes), `noslop_benchmark_results.json`, `noslop_logreg_model.json` (manifest-hashed artifact), `experiments/LABS.md`

---

## 1. Executive summary

We built and validated a two-layer AI-text service: (1) a calibrated binary
AI/human classifier and (2) a slop-pattern diagnosis payload that measurably
improves LLM rewriting. Headline results:

- **Classifier generalization**: multi-domain ensemble reaches **83.8% HC3 /
  86.9% SemEval-OOD** (untouched test sets), vs 54% OOD for single-domain
  training. Domain breadth beats home-turf accuracy.
- **Payload economics**: a **236-token** highlight-shaped diagnostic beats a
  **1,639-token** full-advice payload on both pattern reduction (97% vs 84%)
  and length retention (92.5% vs 73.8%). Less guidance, better outcomes.
- **Surgical patching**: sentence-splice rewrites converge monotonically and
  preserve untouched prose exactly; on a real 1,452-word essay: -45.7%
  patterns at 92.2% retention in 2 iterations.
- **Competitive position**: the slop-linter market is commoditized (7+
  free/open tools). The open ground is agent-native paid rails (x402),
  combined classifier+diagnosis, per-model fingerprint profiles, and a
  continuous tell-discovery pipeline.

---

## 2. Data assets

| Dataset | n | Composition | File |
|---|---|---|---|
| Original set | 55 | literary slop-style AI + human + humanized | `data/test_dataset.json` |
| HC3 | 5,575 | human (3,585) vs ChatGPT (1,990), ≥60 words | `data/benchmarks/hc3_bench.json` |
| SemEval-2024 T8 | 3,600 | human (1,200) vs chatGPT/cohere/davinci (2,400) | `data/benchmarks/semeval_bench.json` |
| Polished v7 essays | 39 | anti-slop-edited AI (adversarial) | `data/benchmarks/polished_ai_bench.json` |

Split discipline (seed-frozen): HC3 humans/ais shuffled with `Random(42)`;
train = first 500/500; held-out = [500:900]; optimizer validation = [900:1300].
SemEval train slice = 500/500 via `hash(text)%10^9` sort + `Random(49)` shuffle;
remainder untouched for OOD testing. Manifest sha256 over all dataset files is
embedded in `miner/src/noslop_logreg_model.json` → `"manifest"` block.

---

## 3. Classifier evolution

### 3.1 Model versions

| Version | Signals | Trainer | Held-out HC3 | SemEval OOD | Artifact |
|---|---|---|---|---|---|
| evolved-linear (v0) | 9 hand-thresholded | GA 150 gens (`evolve_noslop_v2.py`) | 64.9% | ~54% | superseded |
| logreg_v1 | 19 | sklearn LogReg (`train_logreg.py`) | 89.1% [86.8–91.1] | 54.2% ← collapse | `noslop_logreg_model.json` v1 |
| logreg_v3 ensemble | 28 | 2-member weighted ensemble | 86.8% [84.2–88.9] | 84.0% [82.5–85.4] | same file, `"version": "logreg_v3_ensemble_28signals"` |
| **logreg_v4 ensemble (CURRENT)** | **41** | v3 + 13 competitor-derived signals, replace-if-wins promoted | **87.0–88.3%** | **86.5% [85.2–87.8]** | `"version": "logreg_v4_ensemble_41signals"` |

v4 imports (see `COMPETITOR-ANALYSIS.md` §5): staccato-burst, anaphora-abuse,
negation-countdown, colon-elaboration, question-then-answer,
superficial-analysis frames, anchored opener families, vague-attribution,
parenthetical-qualifier, almost-hedge, listicle magic-number (top-4
coefficient at +0.96), sentence-length kurtosis, word-frequency concentration.
Promotion gate: slopscore-style replace-if-wins — must hold HC3 and improve
or hold OOD (it improved both: +2.5 OOD, +3.7 original set).

Key code:
- Feature definitions shared by training and serving: `miner/src/features.py`
  (`FEATURE_FNS`) — v6 pattern counts+rates, focal words (Kobak excess-vocab),
  em-dash rate, connector density, hedge count, reply-bot openers,
  burstiness, LD-score, function words.
- Ensemble inference in pure python (no numpy/sklearn at serve time):
  `miner/src/classifier.py`.
- Training with honest splits + manifest hashing: `train_logreg.py`.

### 3.2 cogym serving-param optimization

Script: `cogym_optimize.py`. Uses `cogym_kernel.evo.recipes`
(`EvolutionContext`, elitist_mutation via `propose_children`) and
`cogym_kernel.eval.gates` (`QualityGate` fail-closed at acc≥0.75 per val
slice; Wilson from gates module).

- Optimizer sees ONLY: hc3[900:1300] + SemEval val slice (300/300).
- Champion: `w_specialist=0.00`, `threshold=0.48` (pure multi-domain member).
- Untouched test: HC3 670/800 = 83.8% [81.0–86.1]; SemEval remainder
  1738/2000 = 86.9% [85.4–88.3].
- Results: `noslop_cogym_optimize_results.json`; serving params written back
  into the model artifact under `"serving_params"`.

### 3.3 Benchmark suite (final config)

Runner: `bench_run.py`; full numbers in `noslop_benchmark_results.json`.

| Set | n | Accuracy | Wilson95 |
|---|---|---|---|
| HC3 full run | 5,575 | 85.9% | [85.0, 86.8] |
| SemEval full (via classify()) | 3,600 | 80.6% | [79.3, 81.8] |
| Polished v7 essays (adversarial) | 39 | ~8% detected | by design: engineered-indistinguishable |
| Original set | 55 | ~78–82% | in-distribution reference |

Latency: p50 ≈ 14ms, p99 ≈ 110ms (in-process); x402 paid round-trip p50 5-13ms.

---

## 4. Deployment verification

Endpoint: `miner/src/x402_server.py` — x402 V1 wire format, Base Sepolia USDC
(`0x036C…CF7e`), $0.003/detect, mock facilitator default
(`MOCK_FACILITATOR=0` switches to live verify/settle).

Tests (all passing):
- Payment flow: `miner/src/x402_client_test.py`
  discovery → unpaid 402+accepts → malformed-payment rejection →
  underpayment rejection → paid 200 + `X-PAYMENT-RESPONSE` settlement →
  20-call latency loop (p50 5ms, spend $0.063).
- Real-agent walkthrough on workstation texts: `agent_demo.py`
  (v7-polished essays verdict=human but 6–18 pattern matches surfaced;
  canonical slop sample verdict=AI conf=1.0; rumi.md human with 4 fixable
  hits — "fixes sloppy human prose too" demonstrated live).

Guidance payload served per request (`guidance` param):
`compact` (default, `glossary_compact` in `miner/src/knowledge.py`) or
`full` (legacy `glossary()`).

---

## 5. Slop-reduction lab (the feedback-loop experiments)

Harness: `slop_loop.py` (single-topic), `lab/payloads.py` (variant ladder),
`lab/ablation.py` (stages, resumable JSONL), `lab/analyze.py` (Wilson +
paired stats). Raw records: `experiments/lab/results.jsonl`; rewritten texts
per episode: `experiments/lab/episodes/`; drafts: `experiments/lab/draft_*.txt`.

Writer/rewriter: `hermes -z` CLI (OpenCode Go; ox-alpha-free quota exhausted →
mimo-v2.5 fallback during runs; noted as environment variable of the experiment).

Baseline drafts are slop-heavy and consistent: love 18 patterns (13 NEG),
time 15 (11 NEG), memory 16 (13 NEG), justice 17 (11 NEG); classifier
confidence < 0.02 (clearly AI). Hermes' unguided prose is dominated by
negation scaffolds — independently confirming the writing corpus's central claim.

### Stage A — payload ablation (full-rewrite mode), n=20 paired episodes

Information ladder: none ⊂ counts ⊂ locate ⊂ highlight ⊂ full.

| arm | tokens* | red% | ret% | clean/4 | paired W/T/L vs none |
|---|---|---|---|---|---|
| none | ~15 | 35.9 | 87.4 | 0 | — |
| counts | ~22 | 82.2 | 90.8 | 0 | 2/0/0 |
| locate | ~180 | 94.0 | 94.2 | 1 | 4/0/0 |
| **highlight** | **~236** | **97.0** | 92.5 | **2** | **4/0/0** |
| full | ~1,639 | 84.3 | **73.8** | 1 | 3/0/1 |

\* measured on draft_love (18 matches): see `lab/payloads.py::build_diagnostic`.

Findings:
1. Every diagnostic arm beats generic control. Diagnosis is the product.
2. **highlight wins outright**: what+where+why at 1/7th the size of `full`.
3. **Over-guiding hurts twice**: `full` reduces less AND guts text
   (retention 73.8%). Scripted repairs/budgets/rules dilute a capable
   rewriter rather than help it.
4. Hermes' dominant tic is NEG scaffolds (11–13 per 450-word essay).

### Stage B — rewrite mode at payload=highlight (n=8)

| mode | red% | ret% | clean/4 | behavior |
|---|---|---|---|---|
| full_rewrite | 92.0 | 89.9 | 1 | rewrites everything; faster convergence |
| patch | 79.0 | 89.4 | 0 | monotonic (18→5, 15→2, 16→5, 17→2); untouched prose preserved byte-exactly |

Patch quality check (love): diff shows only flagged sentences changed;
e.g. "…as though love were a place we wander into, not a decision" →
"…never a choice" (NEG scaffold removed, rhythm kept).

### Large-doc validation (highlight + patch)

Source: `projects/workengestation/output/essays/tantra/tantraloka-essays/daimon-contact.md`
(1,452 words, v7-polished AI, 35 baseline patterns).

| iter | patterns | words |
|---|---|---|
| 0 | 35 | 1452 |
| 1 | 25 | 1430 |
| 2 | 19 | 1339 |

-45.7% patterns at 92.2% retention in 2 iterations; trajectory monotonic —
more iterations available at linear token cost. Output:
`experiments/lab/large_final.txt`. This validates the large-doc architecture:
diagnose globally, patch surgically, iterate to convergence.

### Threats to validity

- Single writer-model (mimo-v2.5 fallback after quota exhaustion); arms were
  paired so comparisons hold within this model, but absolute numbers may vary.
- Reduction % is mechanical (detector-based); semantic quality not scored
  (generator/judge separation respected — no self-grading).
- n=4 per arm; clean-rate CIs are wide. Directional findings are consistent
  across three independent measures (reduction, retention, paired wins).

---

## 6. Competitive landscape (GitHub / commercial review)

Surveyed Aug 2026 via web search + GitHub API. The regex-linter space is
commoditized; none of these are agent-native paid endpoints.

| Tool | Form | Scale | Authorship stance | Rewrite loop | Monetization |
|---|---|---|---|---|---|
| yasyf/slop-cop | Go CLI + Claude/Cursor plugin | 226 rules (178 local + Haiku/Sonnet tiers) | disclaims | yes — agent revises `violations[]` until empty; `rewrite` via claude CLI | OSS (MIT); rides Claude subscription |
| SlopSift | npm linter, dependency parser | rule packs (`ai-style`, `reader-first`) | disclaims | rerun-after-edit guidance | OSS/freemium |
| Slop Sentry | hosted MCP SaaS | 400+ checks + substance/AP-style engines | disclaims | rewrite-until-CLEAN multi-engine pass | **subscription API key** — proof of willingness to pay |
| jman4162/slopscore | Python linter | 16 dimensions, 0–100 score, evidence spans | aggressively disclaims ("never accuse") | `--suggest` advisory only | OSS |
| munzzyy/unslop | single-file CLI, 16 languages | word lists + rhythm CV math | disclaims | hints only | OSS |
| hamelsmu/dslop | CLI for codebases | prose-in-code focus | disclaims | no | OSS |
| awnist/slop-cop | browser editor | 36 rules + 2 LLM passes | disclaims | inline diffs, apply button | OSS |
| GPTZero/Pangram etc. | cloud detectors | statistical classifiers | binary verdicts | none | paid API |

Sources: GitHub repos (yasyf/slop-cop ★9, updated 2026-08-24; jman4162/slopscore
★1), slopsentry.ai, slopsift.dev, munzzyy.github.io/unslop.

### Where NoSlop differs

1. **Paid machine-to-machine rails**: nobody ships x402 micropayments. We do
   ($0.003/call, settle-on-chain receipts). Telegraph hackathon distribution.
2. **Combined calibrated classifier + diagnosis**: linters explicitly refuse
   authorship; detectors give verdicts without actionable output. We serve both
   from one endpoint.
3. **Evidence-based payload design**: Stage A/B methodology (slop-reduction
   per token of guidance) is, to our knowledge, unpublished anywhere. It turns
   prompt design into measurement.
4. **Continuous tell-discovery moat**: literature (Kobak et al. excess vocab;
   locuslab/llm-idiosyncrasies arXiv:2502.12150 — 97.1% five-way model
   attribution from word distributions; Bitton et al. arXiv:2503.01659 —
   0.9988 precision family attribution surviving style prompts) establishes
   that per-model fingerprints are stable and extractable. A pipeline that
   generates fixed prompts on each new frontier model, frequency-diffs against
   human baselines, and ships validated rule updates (gated by our
   slop-reduction episodes) is a service static linters structurally cannot
   match. Per-model products ("NoSlop-GPT/Claude/Grok profiles",
   "which-model-wrote-this" premium signal) follow directly.

Risks: linter features commoditize fast (unslop covers 16 languages free);
differentiation must stay on rails (x402/Telegraph), combined signals,
and update velocity — not on any individual regex list.

---

## 7. Recommended next actions

1. Flip remaining serving defaults to compact guidance (done for endpoints;
   keep `full` available as opt-in for weak models).
2. Register Track 1 miner YAML (real wallet, live facilitator smoke test).
3. Build tell-discovery pipeline in `lab/discover.py`: fixed prompt battery →
   frequency diff vs human baseline → candidate tells → cogym-gated promotion
   into `knowledge.py`/`AI_FOCAL_WORDS` with provenance.
4. Extend lab to second writer-model (cross-family replication per AGENTS.md
   standards: ≥2 families for REPLICATED status).
5. Wire benchmark runs through cogym AsyncRunner → RunReceipts →
   CapabilityClaims for tamper-evident public quality history.

---

## Appendix: artifact map

```
/root/noslop/
├── miner/src/
│   ├── classifier.py                  pure-python ensemble inference
│   ├── noslop_logreg_model.json       frozen model + manifest + serving_params
│   ├── detector.py                    v6 pattern engine (span-deduped)
│   ├── knowledge.py                   pattern KB + glossary()/glossary_compact()
│   ├── features.py                    FEATURE_FNS (28 signals)
│   ├── server.py                      free HTTP endpoint
│   ├── x402_server.py                 x402-gated endpoint
│   └── x402_client_test.py            payment-flow test (ALL PASS)
├── train_logreg.py                    ensemble training + manifests
├── cogym_optimize.py                  kernel-driven serving-param search
├── bench_run.py                       benchmark suite
├── evolve_binary.py / evolve_heldout.py   legacy GA lineage
├── slop_loop.py                       write->diagnose->rewrite assessor
├── agent_demo.py                      paid-endpoint walkthrough
├── data/
│   ├── test_dataset.json              original 55
│   ├── make_split.py                  seed-frozen splitter
│   └── benchmarks/{hc3,semeval,polished_ai}_bench.json
├── lab/
│   ├── payloads.py                    variant ladder builders
│   ├── ablation.py                    episode runner (stages draft/A/B/large)
│   └── analyze.py                     Wilson + paired analysis
├── experiments/
│   ├── LABS.md                        running experiment log
│   └── lab/                           results.jsonl, drafts, episodes/, logs
├── noslop_logreg_model.json           training-time copy
├── noslop_benchmark_results.json      final benchmark numbers
├── noslop_cogym_optimize_results.json champion + untouched-test metrics
└── PROJECT-STATUS.md                  project state overview
```
