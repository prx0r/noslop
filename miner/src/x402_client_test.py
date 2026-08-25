#!/usr/bin/env python3
"""End-to-end x402 client test: simulates an AI agent paying for NoSlop detection.

Flow exercised:
  1. GET /.well-known/x402          (discovery, free)
  2. POST /v1/detect without X-PAYMENT      -> 402 + accepts[]
  3. POST with malformed X-PAYMENT          -> 402 invalid encoding
  4. POST with underpayment                 -> 402 insufficient amount
  5. POST with valid payment                -> 200 + result + X-PAYMENT-RESPONSE
  6. Latency of 20 paid calls

Run while x402_server.py is up on :8090.
"""
import base64
import json
import statistics
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8090"
PAY_TO = "0x742d35Cc6634C0532925a3b844Bc9e7595f5b9Ae"
ASSET = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
NETWORK = "eip155:84532"
PAYER = "0x1234567890AbCdEf1234567890aBcDeF12345678"

SLOP_TEXT = ("The paper opens with alchemy, and the choice is deliberate. "
             "Not conceptual analysis, but participatory knowledge. "
             "Gods, angels, and daimons appear throughout. "
             "We must trust the process and hold space for transformation.")


def post(path: str, body: dict | None = None, payment: dict | None = None):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    if payment is not None:
        raw = payment if isinstance(payment, str) else json.dumps(payment)
        req.add_header("X-PAYMENT", base64.b64encode(raw.encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.load(r), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), dict(e.headers)
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")}, dict(e.headers)


def make_payment(value: str = "3000") -> dict:
    return {
        "x402Version": 1,
        "scheme": "exact",
        "network": NETWORK,
        "resource": f"{BASE}/v1/detect",
        "payload": {
            "signature": "0x" + "ab" * 65,
            "authorization": {
                "from": PAYER, "to": PAY_TO, "value": value,
                "validAfter": int(time.time()) - 60,
                "validBefore": int(time.time()) + 600,
                "nonce": "0x1111111111111111111111111111111111111111111111111111111111111111",
            },
        },
    }


def main():
    print("=== NoSlop x402 end-to-end client test ===\n")

    with urllib.request.urlopen(BASE + "/.well-known/x402", timeout=10) as r:
        disc = json.load(r)
    print(f"1. discovery       OK  price={int(disc['accepts'][0]['maxAmountRequired'])/1e6} USD")

    code, body, _ = post("/v1/detect", {"text": SLOP_TEXT})
    assert code == 402, f"expected 402, got {code}"
    reqs = body["accepts"][0]
    print(f"2. unpaid request  402 '{body['error']}' payTo={reqs['payTo'][:10]}... "
          f"amount={int(reqs['maxAmountRequired'])/1e6} USD")

    code, body, _ = post("/v1/detect", {"text": SLOP_TEXT}, payment="!!!not-base64-json{{{")
    assert code == 402 and "encoding" in body["error"], body
    print(f"3. malformed pmt   402 '{body['error']}'")

    code, body, _ = post("/v1/detect", {"text": SLOP_TEXT}, make_payment("1000"))
    assert code == 402 and "insufficient" in body["error"], body
    print(f"4. underpayment    402 '{body['error']}'")

    t0 = time.perf_counter()
    code, body, headers = post("/v1/detect", {
        "text": SLOP_TEXT, "regime": "short"}, make_payment())
    dt = (time.perf_counter() - t0) * 1000
    assert code == 200, f"{code}: {body}"
    settle = json.loads(base64.b64decode(headers["X-PAYMENT-RESPONSE"]))
    print(f"5. paid request    200 in {dt:.0f}ms  answer={body['answer']} "
          f"conf={body['confidence']:.2f} matches={len(body['matches'])}")
    for m in body["matches"]:
        print(f"     [{m['tag']}] {m['matched_text']!r}")
    print(f"   settlement: tx={settle['transaction'][:18]}... payer={settle['payer'][:10]}... "
          f"success={settle['success']}")

    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        code, _, _ = post("/v1/detect", {"text": SLOP_TEXT}, make_payment())
        assert code == 200
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"\n6. 20 paid calls   mean={statistics.mean(lat):.0f}ms "
          f"p50={sorted(lat)[10]:.0f}ms max={max(lat):.0f}ms")

    total_cost = 21 * 3000 / 1e6
    print(f"\nALL FLOWS PASS. spend=${total_cost:.4f} across 21 paid calls")


if __name__ == "__main__":
    main()
