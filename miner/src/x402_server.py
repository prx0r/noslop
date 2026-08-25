#!/usr/bin/env python3
"""NoSlop x402-gated detection endpoint.

Implements the x402 HTTP-402 payment protocol (V1 wire format):
  - Unpaid POST /v1/detect        -> 402 + accepts[] payment requirements
  - Paid   POST /v1/detect        -> verify -> settle -> 200 + X-PAYMENT-RESPONSE
  - GET    /health                -> free

Facilitator modes:
  MOCK_FACILITATOR=1 (default)  deterministic structural verification, no chain
  MOCK_FACILITATOR=0            calls the live facilitator at FACILITATOR_URL

Conventions follow repos/cogym + 402arena: Base Sepolia USDC, amounts are
6-decimal strings ("3000" = $0.003).

Run: PORT=8090 python3 x402_server.py
"""
import base64
import json
import os
import sys
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from miner.src.classifier import classify
from miner.src.detector import detect
from miner.src.knowledge import glossary, glossary_compact

from server import TAG_TO_KNOWLEDGE

PORT = int(os.environ.get("PORT", 8090))
PRICE = os.environ.get("NOSLOP_PRICE", "3000")  # $0.003 per detection
PAY_TO = os.environ.get("NOSLOP_PAY_TO", "0x742d35Cc6634C0532925a3b844Bc9e7595f5b9Ae")
ASSET = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"  # Base Sepolia USDC
NETWORK = "eip155:84532"
RESOURCE = f"http://localhost:{PORT}/v1/detect"
MOCK = os.environ.get("MOCK_FACILITATOR", "1") == "1"
FACILITATOR = os.environ.get("FACILITATOR_URL", "https://x402.org/facilitator")


def _requirements() -> dict:
    return {
        "scheme": "exact",
        "network": NETWORK,
        "maxAmountRequired": PRICE,
        "resource": RESOURCE,
        "description": "NoSlop AI-text detection with agent fix guidance",
        "mimeType": "application/json",
        "payTo": PAY_TO,
        "asset": ASSET,
        "maxTimeoutSeconds": 30,
        "extra": {"name": "USDC", "version": "2"},
    }


def _decode_payment(header: str) -> dict | None:
    try:
        return json.loads(base64.b64decode(header))
    except Exception:
        return None


def verify_and_settle(payment: dict) -> tuple[bool, str, dict]:
    """Returns (ok, error, settlement)."""
    payload = payment.get("payload", {})
    sig = payload.get("signature")
    auth = payload.get("authorization", {})

    def fail(msg):
        return False, msg, {}

    if payment.get("x402Version") != 1:
        return fail("unsupported x402Version")
    if payment.get("scheme") != "exact" or payment.get("network") != NETWORK:
        return fail("scheme/network mismatch")
    if not isinstance(sig, str) or len(sig) < 16:
        return fail("missing signature")
    if auth.get("to") != PAY_TO:
        return fail("wrong payTo")
    try:
        if int(auth.get("value", "0")) < int(PRICE):
            return fail("insufficient payment amount")
    except (TypeError, ValueError):
        return fail("malformed value")

    if MOCK:
        tx = "0x" + __import__("hashlib").sha256(
            json.dumps(payment, sort_keys=True).encode()).hexdigest()
        return True, "", {
            "success": True, "errorReason": None, "transaction": tx,
            "network": NETWORK, "payer": auth.get("from", "unknown"),
        }

    req = urllib.request.Request(
        f"{FACILITATOR}/verify",
        data=json.dumps({"x402Version": 1, "paymentHeader": base64.b64encode(
            json.dumps(payment).encode()).decode(), "paymentRequirements": _requirements()}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        vres = json.load(r)
    if not vres.get("isValid"):
        return False, vres.get("invalidReason", "verification failed"), {}
    sreq = urllib.request.Request(
        f"{FACILITATOR}/settle",
        data=json.dumps({"x402Version": 1, "paymentHeader": base64.b64encode(
            json.dumps(payment).encode()).decode(), "paymentRequirements": _requirements()}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(sreq, timeout=20) as r:
        sres = json.load(r)
    ok = sres.get("success", False)
    return ok, ("" if ok else sres.get("errorReason", "settlement failed")), sres


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/"):
            self._respond(200, {
                "status": "ok", "miner": "noslop", "intent": "AI_TEXT_DETECTION",
                "method": "v6 patterns + statistical features + fix guidance",
                "payment": {"scheme": "exact", "network": NETWORK,
                            "price_usd": int(PRICE) / 1e6, "asset": ASSET},
                "facilitator_mode": "mock" if MOCK else "live",
            })
        elif self.path == "/.well-known/x402":
            self._respond(200, {"resource": RESOURCE, "accepts": [_requirements()]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/detect":
            self.send_response(404)
            self.end_headers()
            return

        header = self.headers.get("X-PAYMENT")
        if not header:
            self._respond(402, {
                "x402Version": 1,
                "error": "X-PAYMENT header is required",
                "accepts": [_requirements()],
            })
            return

        payment = _decode_payment(header)
        if payment is None:
            self._respond(402, {
                "x402Version": 1,
                "error": "invalid X-PAYMENT encoding (expected base64 JSON)",
                "accepts": [_requirements()],
            })
            return

        ok, err, settlement = verify_and_settle(payment)
        if not ok:
            self._respond(402, {
                "x402Version": 1,
                "error": f"payment rejected: {err}",
                "accepts": [_requirements()],
            })
            return

        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        envelope = body.get("payload", body)
        text = envelope.get("text", "")
        if not text:
            self._respond(400, {"error": "No text provided"})
            return

        regime = envelope.get("regime", "short")
        cls = classify(text)
        det = detect(text, source_text=envelope.get("source_text"))

        seen_tags = sorted({m["tag"] for m in det.details})
        matches = []
        for m in det.details:
            entry = {"tag": m["tag"], "line": m["line"], "matched_text": m["text"]}
            k = glossary([TAG_TO_KNOWLEDGE.get(m["tag"], m["tag"])], regime)["patterns"]
            known = k.get(TAG_TO_KNOWLEDGE.get(m["tag"], m["tag"]))
            if known:
                entry["pattern_name"] = known["name"]
                entry["why_slop"] = known["why_slop"]
                entry["suggested_repairs"] = [r["hint"] for r in known["repairs"][:3]]
            matches.append(entry)

        resp_body = json.dumps({
            "answer": cls["answer"],
            "status": "success",
            "confidence": cls["confidence"],
            "model": cls["model"],
            "matches": matches,
            "guidance": (glossary_compact(det.details) if envelope.get("guidance", "compact") == "compact" else glossary(seen_tags, regime)) if seen_tags else None,
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-PAYMENT-RESPONSE", base64.b64encode(
            json.dumps(settlement).encode()).decode())
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _respond(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"NoSlop x402 endpoint on :{PORT} "
          f"(price ${int(PRICE)/1e6}, facilitator={'mock' if MOCK else 'live'})")
    server.serve_forever()
