# Peer Review: NoSlop Miner vs arXiv Frontier

Date: 2026-08-25
Subject: NoSlop logreg_v4 miner (`EXPERIMENTAL-REPORT.md`, this repo)
Review basis: arXiv/LREC frontier literature (2024–2026) + our internal lab evidence
Stance: hostile-but-fair internal review, written as if an anonymous reviewer
         were grading our submission against current publication standards.

---

## Summary of submitted work

A two-layer AI-text service: (1) a calibrated logistic-regression ensemble
over 41 stylometric/structural signals for binary AI/human classification,
trained multi-domain with frozen splits and fail-closed promotion gates;
(2) a regex-based slop-pattern diagnosis engine emitting agent-facing repair
guidance whose *utility* is optimized via a measured write→diagnose→rewrite
loop. Deployed behind an x402-paid HTTP endpoint.

---

## Major concern 1: We report accuracy@threshold where the frontier reports
## threshold-free and low-FPR metrics

**Frontier standard:** the comprehensive benchmark (arXiv:2603.17522) applies
a five-metric suite — AUROC, AUPRC, EER, Brier, FPR@95%TPR. Stowe & Patil
(arXiv:2604.16607 / LREC 2026) show model *rankings* flip depending on metric
choice and threshold placement, and demand justified, multi-metric evaluation.
The Waterloo generalizability study goes further: "commonly reported metrics
such as AUROC can obscure poor performance at deployment-relevant thresholds"
— detectors with high AUROC yield near-zero TPR at low FPR.

**Our exposure:** we report accuracy at one operating point plus Wilson CIs.
We never computed AUROC despite our logreg emitting continuous probabilities.
An FP on a human writer is the high-consequence error in every deployment
story that matters (moderation, hiring, academia); without TPR@FPR-1% we
cannot say how we behave there.

**Verdict:** VALID CRITICISM. Cheap to fix: our probabilities exist; add
AUROC/AUPRC/Brier/EER-thresholded accuracy/TPR@FPR-1% to `bench_run.py`.

---

## Major concern 2: The length confound is half-addressed

**Frontier:** arXiv:2603.17522 applies principled ±20% length-matching before
splitting, citing that without it "classical detectors trivially exploit the
length disparity." Pudasaini et al. (arXiv:2603.23146) find misclassified
texts are systematically shorter (FP modal length 34 words) and recommend
minimum-viable-length thresholds or length-aware normalization.

**Our exposure:** we found and fixed the raw-counts-vs-rates bias reactively
(demo text scored human until pattern features became rates), but our eval
sets are NOT length-matched, we don't report performance by length bucket,
and the endpoint happily classifies 14-word inputs whose features are noise.
Our HC3 filter (≥60 words) quietly excludes the hardest regime.

**Verdict:** VALID. Partially mitigated (rate features), but reviewers would
demand length-bucketed reporting and a documented short-text policy
(we should return abstention or wide-interval confidence below ~50 words).

---

## Major concern 3: Generator coverage stops in 2023

**Frontier:** arXiv:2603.17522 evaluates zero-shot against five unseen
open-source LLMs (TinyLlama → Llama-3.1) plus L0–L2 adversarial humanization;
the survey record treats cross-generator shift as the core open problem.
Fingerprint work (arXiv:2502.12150, 2503.01659) shows idiosyncrasies are so
model-specific that attribution among GPT/Claude/Grok/Gemini/DeepSeek hits
97% — implying binary detectors tuned on ChatGPT-era text may silently miss
modern generators.

**Our exposure:** HC3 = ChatGPT only. SemEval adds cohere/davinci (older
still). Our only modern-generator evidence is 39 v7-edited essays where we
detect ~10%. We have zero Claude-4/Gemini/Llama-3/GPT-5-era samples. Our OOD
numbers are honest *for the generator families tested* and say nothing about
2026 models.

**Verdict:** VALID and the most important scientific gap. Mitigation is the
planned tell-discovery pipeline + periodic re-acquisition of fresh-generator
benchmarks; also aligns with Waterloo's conclusion that robust detection
"requires continually recalibrated, model-aware pipelines rather than static
universal detectors" — which is literally our cogym loop, but it needs fresh
data to chew on.

---

## Major concern 4: Dataset-artifact reliance is unexamined

**Frontier:** Pudasaini et al.'s SHAP analysis shows top features differ
across corpora — detectors learn corpus cues, not authorship. Their concrete
embarrassment: a classifier keying on paragraph count (TP median 1 vs TN
median 17).

**Our exposure:** our blended top coefficients are ngram_dens (+1.44),
neg (−1.21!), vocab_div (−1.05), listicle (+0.96). Two red flags a reviewer
would poke: (a) NEG's coefficient is *negative* — more negation scaffolds
push toward "human" in the blend, which contradicts our own corpus doctrine
and suggests the members learned opposite signatures that partially cancel;
(b) vocab_div and fw_ratio may encode domain formality rather than
machine-ness. We never ran SHAP/permutation importance per member, nor
compared feature stability across training slices — exactly the check
arXiv:2603.23146 performs.

**Verdict:** VALID. Fix: permutation-importance table per member per dataset;
investigate the NEG sign inversion before shipping any claim about mechanism.

---

## Major concern 5: Calibration asserted, never measured

**Frontier:** Brier is in the five-metric suite; arXiv:2602.08031 builds an
entire paper around detection-score calibration (Markov-informed correction);
the Waterloo work calibrates thresholds on human-only pools to make FPR
statistically meaningful at 0.1%.

**Our exposure:** we say "calibrated confidence" because a sigmoid produces
numbers in [0,1], but logistic outputs from a StandardScaler pipeline are not
calibrated probabilities. No ECE/Brier/reliability curve exists in the repo.
If an agent gates actions on conf > 0.9, we don't know what that means.

**Verdict:** VALID. Cheap: reliability diagrams + Brier on held-out sets;
Platt/isotonic rescaling if curves show distortion; adopt the human-only-pool
threshold-calibration idea for a promised FPR level.

---

## Major concern 6: Evaluation-set contamination in the "combined" number

Our own `bench_run.py` COMBINED row includes the 1,165 training entries
(HC3 train slice + original×3). Small relative weight (~20%), and the clean
held-out numbers are reported separately — but a reviewer would flag that
88.3%-full-run headlines sit next to 87.0% held-out with identical ordering,
inviting over-claim. The report already separates them; keep it that way and
never quote combined as headline.

**Verdict:** MINOR but reputational. Add an explicit "train entries excluded"
variant to bench output.

## Major concern 7: Notion of detection unspecified

Dycke et al. (arXiv:2606.04906) show AI-detection datasets bake in hidden
notions (fully-generated vs co-written vs policy-based) and that these notions
change both difficulty and false-positive behavior on human-AI collaborative
text. Our own original set contains 'mixed' entries (n=1, scored wrong).

**Exposure:** our spec never states we detect *document-level, fully-machine-
generated* text. Co-written prose will drift toward whichever signal dominates.
**Verdict:** VALID. One paragraph in the spec + a `notion` field in the API
response costs nothing and preempts the criticism.

## Minor concerns

- Multiple comparisons: Stage A ran 5 arms × 3 metrics; doctrine requires
  correction at >5 variants. Borderline; note Bonferroni-safe interpretation
  (highlight wins by margins larger than any correction).
- Single-writer-model lab (mimo fallback): paired design protects within-run
  comparisons; absolute reduction numbers need replication on a second family
  (AGENTS.md REPLICATED bar: ≥2 families).
- Perplexity-family signals absent: sentence-level perplexity CV is the
  single most discriminative feature in arXiv:2603.17522's stylometric
  hybrid. We deliberately avoid LM inference cost — fine for deployment, but
  the report should name this as a known ceiling, and quantify it someday via
  a small distilled scorer.
- Baselines: we compare against our own lineage, not RoBERTa/Binoculars/
  Fast-DetectGPT runs on our splits. Frontier baseline-first work says
  fine-tuned RoBERTa ≈ everything in-distribution; our defensible claim is
  the OOD/deployment/cost trade-off, which needs their numbers on OUR splits
  to be credible.
- Human-only calibration pool (Waterloo) would let us promise "≤1% FPR on
  human text" — currently our FPR is whatever the threshold happens to give.

## Where the work is AHEAD of the frontier

Credit where the reviewer must concede:

1. **Agent-utility optimization is novel.** No paper evaluates a detector by
   downstream repair utility (our Stage A/B slop-reduction episodes). The
   closest published idea is instance-level explanation packages
   (arXiv:2603.23146 releases prediction+explanation) — but none measures
   whether explanations improve a rewriter, and none discovers that less
   guidance wins.
2. **Distribution-shift response matches the frontier's prescription.**
   2607.03680 concludes progress should be measured under shift; Waterloo
   demands continually recalibrated model-aware pipelines. Our multi-domain
   ensemble + cogym re-tuning loop is exactly that shape, with promotion
   gates slopscore-style.
3. **Interpretability is native, not post-hoc.** Coefficients + pattern
   evidence + disguise catalogs answer the XAI demand for transparent
   cues without SHAP overhead.
4. **Reproducibility discipline exceeds publication norms:** frozen seeds,
   manifest sha256 over datasets+params, resumable JSONL episode logs,
   fail-closed gates. Most arXiv papers release none of this.
5. **Honest negative results retained:** polished-AI ~10%, single-domain OOD
   collapse 54% — reported, not buried; consistent with Pudasaini et al.'s
   "benchmark accuracy is not reliable evidence" stance.
6. **Deployment context justification** (Stowe's requirement) is implicit in
   the product: FP cost is low (diagnosis still useful), FN cost is low
   (human-slop gets fixed anyway) — the metric mix we chose fits the use case,
   we just need to SAY so explicitly.

## Required revisions (prioritized)

| # | Revision | Effort | Addresses |
|---|---|---|---|
| R1 | Add AUROC/AUPRC/Brier/EER-acc/TPR@FPR1% to bench_run.py | hours | MC1 |
| R2 | Length-bucketed eval + <50-word abstention policy | hours | MC2, Pudasaini |
| R3 | Reliability diagram + Platt scaling; human-pool threshold calibration promising ≤1% FPR | day | MC5, Waterloo |
| R4 | Permutation importance per member; explain NEG sign inversion | day | MC4 |
| R5 | Acquire ≥2 modern-generator benchmark slices (Claude/Gemini/Llama-3 era); add to OOD rotation | days | MC3 |
| R6 | State document-level fully-generated notion in spec + API `notion` field | hour | MC7 |
| R7 | Run RoBERTa/Binoculars baseline on our splits for the comparison table | day | Minors |
| R8 | Replicate Stage A on second writer-model family | days | Minors |

## Review verdict

**Major revision** — scientifically sound core with unusually good
reproducibility discipline and one genuinely novel contribution (measured
agent-utility optimization), but evaluated below current metric standards
(no AUROC/low-FPR/calibration), with stale generator coverage and unexamined
artifact reliance. All required revisions are executable within days using
existing machinery; none threatens the architecture. R1–R4 should land before
any external claim beyond "internal prototype."

---

## Revision log

### R1 EXECUTED — five-metric suite (`eval_metrics.py`, `noslop_frontier_metrics.json`)
Held-out HC3 (n=800): **AUROC 0.9462, AUPRC 0.9429, Brier 0.0938,
acc@EER-threshold 86.75%** — competitive with the stylometric-hybrid family
in arXiv:2603.17522. The frontier's warning confirmed quantitatively:
raw TPR@FPR-1% = 33.8%. High AUROC, weak low-FPR operation — exactly the
Waterloo failure shape.

### R2 PARTIAL — length buckets measured; abstention shipped
Accuracy rises monotonically with length: 84.8% (50–150w) → 87.8%
(150–300w) → 92.7% (300w+). Serving now flags `reliable:false` below 50
words with an explanatory note (`classifier.py`); full abstention policy
still open.

### R3 EXECUTED — Platt calibration + human-pool FPR promise (`calibrate.py`)
Platt fit on untouched slice [1300:1800]; decision threshold set at the 99th
percentile of scores on a **1,785-document human-only pool** (humans[1800:]).
Result: promised ~1% FPR achieved on pool (1.01%), holds at 1.50% on unseen
human holdout; **TPR at that operating point: 47.5% HC3 / 72.0% SemEval**
(up from 33.8% pre-calibration). Calibration block stored in the model
artifact; serving inference applies it (pure-python parity verified to 5e-5).

### R4 EXECUTED — NEG sign inversion explained (`noslop_frontier_metrics.json`)
Per-member audit: hc3_specialist neg coef −0.18 (mild), multidomain −1.65
(strong). In SemEval's wiki/Reddit human prose, "not X but Y" scaffolds are
*human-common*, so the multi-domain member legitimately learned NEG→human.
The ensemble blend inherits the negative sign. Not a bug: the classifier and
the diagnosis layer answer different questions — the classifier measures
domain-conditional likelihood, the diagnosis layer flags register-relative
craft patterns regardless of authorship. Documented tension; both layers are
serving their own purpose. Permutation importances captured per member
(top: fw_ratio, three_list_rate, burs, ngram_dens).
