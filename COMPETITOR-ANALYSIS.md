# NoSlop — Competitor Analysis

Date: 2026-08-25
Clones inspected: `/root/competitors/{slop-cop,slop-cop-browser,slop-lint,slop-gate,slopscore,slopsift,dslop,unslop-mshumer}`
Companion docs: `EXPERIMENTAL-REPORT.md` §6, `experiments/LABS.md`

---

## 1. The field at a glance

| Repo | Language | Scale | Distinctive move | Stars |
|---|---|---|---|---|
| yasyf/slop-cop | Go | 226 rules (48 slop + 9 base + 169 google; 178 client + 40 Haiku + 8 Sonnet) | JSON rap sheet agents revise against; tree-sitter masking; span-scoped supersession | 9 |
| Slop Sentry | hosted MCP SaaS | 400+ checks + substance/hokiness/AP-style engines | rewrite-until-CLEAN multi-engine editorial pass | n/a (subscription) |
| mshumer/unslop | Python+Claude Code | per-domain | **tell-discovery pipeline**: generate samples → measure repeated defaults → emit avoidance skill | 529 |
| jman4162/slopscore | Python | 16 dimensions, 0–100 score | evidence spans + published fairness audit (per-rule FP rate on plain/ESL English) | 1 |
| eric-sabe/slop-lint | Node | ~100 words/phrases | em-dash = hard fail; `--discover` frequency mining; sourced+versioned catalog | small |
| hwajongpark/slop-gate | Node | en + ko/ru/vi/zh/fil packs | translationese framing per language; CI exit codes | small |
| NikhilVerma/writinglint (SlopSift) | TS/npm | dependency-parser rules | grammatical-relation matching, exact ranges, local-first | small |
| hamelsmu/dslop | Rust | prose-in-code | statistical metrics: sentence-length CV/kurtosis, word-freq dispersion | small |
| awnist/slop-cop | browser TS | 36 rules + 2 LLM passes | real-time highlight editor | small |

## 2. Deep dive: yasyf/slop-cop (closest structural competitor)

### Architecture (`internal/detectors/`, `cmd/slop-cop/check.go`)
Length-preserving masking first (goldmark AST for markdown, tree-sitter for
JS/TS, `x/net/html` for HTML) so `delve` inside code fences never fires;
then pure precompiled regex + counters on prose; optional Claude tiers ride
the user's own `claude` CLI with schema-forced JSON; merge dedupes by
(ruleId,start,end); google-layer hits supersede looser rules **span-locally**.

### Their 48 slop rules vs our signals

**They have, we lack (cheap asymmetric imports):**
1. Cross-sentence rhythm crimes: `staccato-burst` (≥3 sentences ≤8 words),
   `anaphora-abuse` (≥3 shared openers), `gerund-litany`,
   `negation-countdown` (≥2 consecutive "Not " starts). Our FLAT only
   measures length variance — none of these shapes.
2. Shape rules: `colon-elaboration` (short-clause:long-elaboration),
   `question-then-answer`, `superficial-analysis` (", highlighting its
   importance," trailing-participle frame).
3. Opener families anchored at sentence start: `era-opener`, `imagine-world`,
   `heres-the-kicker`, `false-conclusion`. Our AI_NGRAMS catches some by
   substring without anchoring.
4. Epistemic hygiene: `vague-attribution` ("studies show"), `almost-hedge`,
   `parenthetical-qualifier`, `unnecessary-contrast`, `broader-implications`,
   `concept-label` ("the [x] paradox").
5. Markdown-structure tells: `listicle-instinct` (lists of exactly 3/5/7/10),
   `bold-first-bullets`, `dramatic-fragment`, `unicode-arrows`.
6. Infra worth copying: byte-offset spans (we return line numbers only),
   input masking (our regexes would false-positive inside code blocks).

**We have, they structurally lack:**
1. **NARR** — academic narration tics ("The paper opens with", "This section
   examines"). Invisible to all 226 of their rules; doc-tier
   `fractal-summaries` is far narrower.
2. **Source-relative checks** — PARA (paraphrase-itis, ≥2:1 analysis ratio)
   and SAME-ENDING (closing echoing source). They are purely intrinsic.
3. **A trained, calibrated classifier** — LD-score, burstiness, function-word
   stats, abstract/concrete density, logreg confidence. They have no score at
   all ("readability is never a gate").
4. **Regime budgets & repair epistemology** — short=zero-tolerance vs
   long=purpose-tested; semantic-inversion warning; repair-tool rotation;
   positive targets. Their guidance says what to remove; ours encodes *when a
   pattern is permitted* and what to build instead.
5. **Self-help register clichés** (power of now, hold space) — their
   `metaphor-crutch` is business-speak only.

### Their agent-loop contract (`skills/slop-cop-prose/SKILL.md`)
Draft → check → revise against violations[] → re-check; fix priority
google>slop>base; **cap 4 passes**, stop when count stops dropping; safe
corridor of 9–40-word sentences; silence discipline ("do not announce the
loop"). Their `rewrite.go` system prompt: 8 always-on directives +
"preserve factual content exactly" meta-principles + per-violation hints.

Comparison to our lab result: their loop ≈ our Stage B full_rewrite mode with
a heavier always-on directive set. Our Stage A finding (highlight-shaped
diagnosis beats scripted directives at 1/7 the tokens) predicts their rewrite
prompt over-guides — testable by running their binary through our harness.

## 3. The others in one paragraph each

- **mshumer/unslop (★529)** — the discovery-pipeline idea as a product:
  generate N samples per domain via Claude Code, cluster repeated defaults,
  emit an avoidance `skill.md`, before/after comparison. Validates our
  planned `lab/discover.py` moat; differs in being a one-shot skill
  generator, not a continuously-updated detection service with paid rails.
- **slopscore** — closest to honest evaluation culture (held-out benchmark,
  PR-AUC 0.91 overt / 0.69 subtle, fairness slices). Its v0.3 lesson mirrors
  ours: a learned scorer must beat the rule scorer under replace-if-wins or
  it doesn't ship — same gate philosophy as our cogym fail-closed promotion.
- **slop-lint** — `--discover` mines over-represented tokens model-vs-baseline
  and model-vs-model-pool (topic-held-constant), requires 2 independent
  sightings before a rule ships, CHANGELOG-versioned catalog. Best-in-class
  provenance hygiene; directly reusable design for our discover pipeline.
- **slop-gate** — per-language packs named after native phenomena
  (Korean 번역투, Russian канцелярит, Chinese 公文腔). Multilingual expansion
  template if we ever internationalize.
- **dslop** — statistical metric gates (sentence-length CV ≤0.3, kurtosis,
  top-n word-frequency dispersion per 200-word chunk) — cheap feature ideas
  for `features.py`.
- **SlopSift** — dependency-parser rules and findings-per-thousand-words
  normalization; also ships an "agent skill" editing procedure like ours.
- **Slop Sentry** — the commercial proof: MCP endpoint, API keys, tiered
  subscription, multi-engine rewrite-until-clean. No x402, no classifier, no
  public evidence of payload optimization.

## 4. Synthesis — where the moat actually is

1. **Rails**: zero competitors on x402/Telegraph. Machine-native distribution
   is ours to take (Track 1 registration + Track 3 app feeding it demand).
2. **Dual signal**: calibrated AI-confidence + actionable diagnosis in one
   call. Linters refuse authorship; detectors refuse usefulness.
3. **Update velocity as a service**: discovery pipelines exist as one-shot
   skills (mshumer) or local tools (slop-lint), but nobody runs a hosted,
   continuously-updated, evidence-gated pattern service. Our cogym harness is
   the gating mechanism they lack.
4. **Payload science**: we can claim (with published numbers) the
   minimum-effective-payload result — 236-token diagnosis beating 1,639-token
   scripted advice on reduction AND retention. Directly contradicts
   slop-cop's heavy-directive design; falsifiable against their binary.

## 5. Immediate imports (hours of work, feeds both layers)

Into `miner/src/features.py` (+ detector rules):
- staccato-burst, anaphora-abuse, gerund-litany, negation-countdown
  (cross-sentence counters over existing sentence splitter)
- colon-elaboration, question-then-answer, superficial-analysis frames
- era-opener/imagine-world/heres-the-kicker/false-conclusion openers,
  sentence-start anchored
- vague-attribution, almost-hedge, parenthetical-qualifier lists
- listicle magic-number counter (markdown-aware line pass)
- dslop-style: sentence-length CV (have), kurtosis, word-freq dispersion

Then re-run `train_logreg.py` + `bench_run.py`; promote new features only if
held-out accuracy holds (replace-if-wins gate, slopscore-style).
