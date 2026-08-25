# NoSlop — AI Text Detection & Slop-Diagnosis Miner

Telegraph Protocol Hackathon — AI_TEXT_DETECTION intent (Track 1) and
Track-3 unslop-agent application. See PROJECT-STATUS.md for current state,
experiments/LABS.md for the live experiment log.

## What it does

NoSlop is two services in one:

1. **Classifier** (`answer`: human=0 / AI=1 + calibrated confidence) —
   `logreg_v3`, an ensemble of two logistic models over 28 stylometric and
   structural signals, trained multi-domain (HC3 + SemEval-2024 T8) with
   honest held-out numbers: ~84% HC3, ~87% SemEval OOD.
2. **Diagnosis payload** — per-line slop-pattern matches (NARR / NEG / 3LIST /
   CLICHE / FLAT) with why-it-reads-as-machine explanations and regime-aware
   repair guidance. This is the product: agents pay to know *what to fix*.

## Quick start

```bash
# paid endpoint (x402, $0.003/detect, Base Sepolia USDC)
python3 miner/src/x402_server.py            # PORT=8090 default

# simulate an agent paying
python3 miner/src/x402_client_test.py       # full flow test, all green
python3 agent_demo.py                       # real workstation texts through paid flow
```

Request:
```json
POST /v1/detect
{"text": "...", "regime": "short|long", "source_text": "(optional)"}
```

Response: `answer`, `confidence`, `matches[]` (tag/line/text/why/repairs),
`guidance` (pattern glossary + budgets + sequencing rules), settlement in
`X-PAYMENT-RESPONSE` header. Mock facilitator by default;
`MOCK_FACILITATOR=0 NOSLOP_PAY_TO=0xYourWallet` for live verify/settle.

## Training / benchmarking

```bash
python3 train_logreg.py        # trains ensemble members -> frozen artifact+manifest
python3 cogym_optimize.py      # cogym-kernel tuning of serving params
python3 bench_run.py           # full benchmark suite w/ Wilson CIs
```

## Slop-reduction lab

```bash
python3 slop_loop.py --topic love --iters 2     # write -> diagnose -> rewrite loop
nohup python3 lab/ablation.py --stage A ... &   # payload ablation (see LABS.md)
python3 lab/analyze.py --stage A                # Wilson CIs + paired wins
```

## Files

| Path | Purpose |
|------|---------|
| miner/src/classifier.py | pure-python ensemble inference |
| miner/src/noslop_logreg_model.json | frozen model + manifest sha256 + serving params |
| miner/src/detector.py | v6 pattern engine (deduped) |
| miner/src/knowledge.py | pattern knowledge base distilled from writing corpus |
| miner/src/features.py | 28 canonical signal functions |
| miner/src/x402_server.py | x402-gated endpoint |
| lab/payloads.py | guidance-payload ladder (none..full) |
| experiments/LABS.md | experiment log with interim results |
| PROJECT-STATUS.md | full state of the project |
