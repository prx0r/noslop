#!/usr/bin/env python3
"""Agent walkthrough: real workstation texts through the paid x402 endpoint.

Simulates exactly what a Telegraph agent does: discovery -> 402 challenge ->
sign payment -> retry -> consume result + fix guidance.

Run while x402_server.py is up on :8090.  python3 agent_demo.py
"""
import base64
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8090"
PAY_TO = "0x742d35Cc6634C0532925a3b844Bc9e7595f5b9Ae"
NETWORK = "eip155:84532"
PAYER = "0x1234567890AbCdEf1234567890aBcDeF12345678"

CASES = [
    ("v7-polished AI essay (workengestation expansion-essay1)",
     "/root/projects/workengestation/output/essays/tantra/tantraloka-essays/expansion-essay1.md",
     4000),
    ("v7-polished AI essay (daimon-contact)",
     "/root/projects/workengestation/output/essays/tantra/tantraloka-essays/daimon-contact.md",
     4000),
    ("AI slop sample (original test set ai_001)",
     None, None),
    ("human text (content-sources rumi.md excerpt)",
     "/root/content-sources/sufism/rumi/rumi.md", 3000),
]

SLOP_SAMPLE = """The paper opens with alchemy, and the choice is deliberate because alchemy
provides the controlling metaphor for everything that follows. Not conceptual analysis,
but participatory knowledge. The author introduces three key concepts: gods, angels,
and daimons. This section examines the role of the active imagination. We now turn to
the question of how symbols participate in divine order. It is important to note that
the power of now plays a crucial role in this transformative journey."""


def load(path, limit):
    t = open(path).read()
    body = "\n".join(l for l in t.split("\n") if not l.strip().startswith("#"))
    return " ".join(body.split())[:limit]


def pay_and_call(text, regime="long"):
    req = urllib.request.Request(BASE + "/v1/detect", method="POST",
                                 data=json.dumps({"text": text, "regime": regime}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r), dict(r.headers)
    except urllib.error.HTTPError as e:
        challenge = json.loads(e.read())
        payment = {
            "x402Version": 1, "scheme": "exact", "network": NETWORK,
            "resource": f"{BASE}/v1/detect",
            "payload": {"signature": "0x" + "cd" * 65,
                        "authorization": {"from": PAYER, "to": PAY_TO,
                                          "value": challenge["accepts"][0]["maxAmountRequired"],
                                          "validAfter": int(time.time()) - 60,
                                          "validBefore": int(time.time()) + 600,
                                          "nonce": "0x22" * 31 + "01"}},
        }
        req2 = urllib.request.Request(BASE + "/v1/detect", method="POST",
                                      data=json.dumps({"text": text, "regime": regime}).encode(),
                                      headers={"Content-Type": "application/json"})
        req2.add_header("X-PAYMENT", base64.b64encode(json.dumps(payment).encode()).decode())
        with urllib.request.urlopen(req2, timeout=30) as r:
            return r.status, json.load(r), dict(r.headers)


def main():
    print("=" * 74)
    print("AGENT WALKTHROUGH: paying $0.003/detection over x402")
    print("=" * 74)

    texts = []
    for label, path, limit in CASES:
        if path:
            texts.append((label, load(path, limit)))
        else:
            texts.append((label, SLOP_SAMPLE))

    # Full raw output for the first case — show EXACTLY what agent receives.
    label, text = texts[0]
    print(f"\n--- FULL RAW RESPONSE: {label} ---")
    code, body, headers = pay_and_call(text)
    slim = {k: v for k, v in body.items() if k != "signals"}
    print(json.dumps(slim, indent=1)[:3500])
    settle = json.loads(base64.b64decode(headers["X-PAYMENT-RESPONSE"]))
    print(f"\nX-PAYMENT-RESPONSE (settlement receipt): tx={settle['transaction'][:20]}... "
          f"payer={settle['payer'][:10]}... success={settle['success']}")

    print(f"\n{'='*74}\nALL CASES\n{'='*74}")
    total_words = 0
    for i, (label, text) in enumerate(texts):
        t0 = time.perf_counter()
        code, body, _ = pay_and_call(text)
        dt = (time.perf_counter() - t0) * 1000
        verdict = "AI" if body["answer"] == 1 else "HUMAN"
        n_match = len(body.get("matches") or [])
        tags = {}
        for m in body.get("matches") or []:
            tags[m["tag"]] = tags.get(m["tag"], 0) + 1
        top_repairs = ""
        if body.get("matches"):
            r0 = body["matches"][0].get("suggested_repairs", [""])
            top_repairs = r0[0][:80]
        print(f"\n{i+1}. {label}")
        print(f"   words={len(text.split())}  ->  {verdict}  conf={body['confidence']} "
              f"({dt:.0f}ms incl. payment)")
        print(f"   pattern matches: {n_match} {tags if tags else ''}")
        if top_repairs:
            print(f"   first repair hint: \"{top_repairs}\"")
        total_words += len(text.split())

    print(f"\ntotal spend: ${len(texts)*0.003:.4f} for {total_words} words audited")


if __name__ == "__main__":
    main()
