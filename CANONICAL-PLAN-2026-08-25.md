# NoSlop — Canonical Plan

Timestamp: 2026-08-25T24:00 UTC (session close)
Supersedes: none (first canonical issue)
Status of record: git-tracked repo, manifest-hashed model artifact
(`miner/src/noslop_logreg_model.json`), episode ledger
(`experiments/lab/results.jsonl`)

This is the single source of truth for what NoSlop is, what has been
established, and what happens next. Later plans must supersede this file by
name and timestamp.

---

## 1. Established facts (do not relitigate)

| ID | Fact | Evidence |
|---|---|---|
| F1 | Binary classifier generalizes: HC3 held-out AUROC 0.946, acc 87–88%; SemEval OOD 86.1% | `experiments/results/noslop_frontier_metrics.json`, `noslop_benchmark_results.json`, `noslop_cogym_optimize_results.json` |
| F2 | Single-domain training collapses OOD (54% → 86% fixed by multi-domain ensemble) | `evolve_binary.py` run vs `train_logreg.py`; PEER-REVIEW MC3 |
| F3 | Minimum-payload law: 236-token highlight diagnosis beats 1,639-token scripted advice on reduction (97% vs 84%) AND retention (92.5% vs 73.8%) | Stage A, 20 paired episodes, `experiments/lab/results.jsonl` |
| F4 | Patch-mode rewriting converges monotonically with byte-exact preservation; large-doc validated -45.7% @ 92.2% retention | Stage B + large run, `experiments/lab/episodes/` |
| F5 | Low-FPR operation quantified: TPR@1%FPR = 47.5% HC3 / 72% SemEval after Platt calibration from a human-only pool | `calibrate.py`, artifact `calibration` block |
| F6 | Competitor imports pass replace-if-wins: 13 signals added, listicle magic-number became top-4 coefficient (+2.5 OOD) | `COMPETITOR-ANALYSIS.md` §5, LABS.md integration lab |
| F7 | Hermes' unguided prose is NEG-slop-saturated (11–13 scaffolds per 450-word essay) | baseline drafts, `experiments/lab/draft_*.txt` |
| F8 | Polished v7-edited AI scores human (~10% detection) — by design boundary, not failure | adversarial benchmark slice |

## 2. Architecture frozen for v1 submission

- **Classifier**: logreg_v4 ensemble (41 signals), serving w=0.25/t=0.58 +
  Platt calibrator; pure-python inference. `miner/src/classifier.py`
- **Diagnosis**: v6 pattern engine + knowledge base; compact highlight-shaped
  guidance default. `miner/src/detector.py`, `miner/src/knowledge.py`
- **Rail**: x402 V1, $0.003/detect, Base Sepolia USDC, mock facilitator
  default → live switch via env. `miner/src/x402_server.py`
- Known boundaries (documented, not hidden): short texts <50 words flagged
  `reliable:false`; edited-AI detection ~10%; generator coverage pre-2024.

## 3. Roadmap

### Phase 0 — SUBMIT (immediate)
- [ ] P0.1 Telegraph Track 1 registration: miner YAML (AI_TEXT_DETECTION),
      real wallet in NOSLOP_PAY_TO, live facilitator smoke test
      (`MOCK_FACILITATOR=0`)
- [ ] P0.2 Public X post per judging requirement (75/25 rule); link demo
- [ ] P0.3 Keep miner live through Track 3 window (rules require uptime)

### Phase 1 — TRACK 2 ENTRY (script author)
- [ ] P1.1 WASM evaluation script for AI_TEXT_DETECTION implementing
      frontier-grade grading: AUROC/AUPRC/Brier/EER-acc/TPR@FPR1%,
      length-bucketed accuracy (from `eval_metrics.py`) — positions us to
      grade incumbents on axes they ignore (humanized recall, calibration)
- [ ] P1.2 Earns recurring revenue per run; also shapes our own leaderboard

### Phase 2 — TRACK 3 ENTRY (application)
- [ ] P2.1 Unslop agent product: detect (Telegraph miner) → rewrite
      (Groq 102 / any LLM miner) → re-check loop until matches empty;
      patch mode for long docs. Skeleton exists (`agent_demo.py`,
      `slop_loop.py`, `lab/payloads.py`)
- [ ] P2.2 Generates ≥100 real requests into our intent (prize guardrail)
- [ ] P2.3 Uses only real miners per rules; every call x402-receipted

### Phase 3 — SCIENCE DEBT (peer-review R5–R8)
- [ ] P3.1 R5: modern-generator benchmarks (Claude/Gemini/Llama-3 era slices);
      add to OOD rotation, retrain if drift detected
- [ ] P3.2 R7: RoBERTa/Binoculars baseline on OUR splits for honest comparison
- [ ] P3.3 R8: replicate Stage A on second writer-model family (AGENTS.md
      REPLICATED bar)
- [ ] P3.4 Full short-text abstention policy (return wide interval below 50w)

### Phase 4 — THE MOAT (continuous evolution service)
- [ ] P4.1 `lab/discover.py`: fixed prompt battery → generate on new frontier
      model → frequency diff vs human baseline (slop-lint's --discover
      method) → candidate tells
- [ ] P4.2 Candidates gated by slop-reduction episodes before promotion into
      `AI_FOCAL_WORDS`/knowledge base, provenance-tagged (slop-lint's
      versioned-catalog discipline)
- [ ] P4.3 Per-model profiles ("NoSlop-GPT/Claude/...") grounded in fingerprint
      literature (arXiv:2502.12150: 97% five-way attribution)
- [ ] P4.4 cogym RunReceipts → CapabilityClaims for public verifiable quality
      history (machinery exists in repos/cogym, unwired)

## 4. Operating rules going forward

1. Every model change passes the replace-if-wins gate on frozen splits
   (heldout HC3 + SemEval OOD) before deployment. No exceptions.
2. Every experiment appends to `experiments/lab/results.jsonl` or a sibling
   ledger with a manifest hash. LABS.md stays current.
3. One variable at a time; paired designs where arms share drafts.
4. Negative results are recorded in LABS.md, never deleted.
5. Serving artifacts change only through train→gate→optimize→deploy chain.
6. Claims about performance quote held-out numbers with Wilson CIs, never
   train-set accuracy; combined-run numbers are never headline.

## 5. File map

See `INDEX.md` for the labelled inventory of every file and directory.
