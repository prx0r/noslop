#!/usr/bin/env python3
"""NoSlop AI Text Detection Miner — Telegraph AI_TEXT_DETECTION endpoint.

Uses cogym-evolved weights for v6 pattern detection + statistical features.
Responses include agent-facing fix guidance from the writing-corpus knowledge base.
"""
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from miner.src.classifier import classify
from miner.src.detector import detect
from miner.src.knowledge import glossary, glossary_compact

# Evolved weights retained for the legacy extract() path; the live classifier
# is miner.src.classifier (logreg_v1).
EVOLVED_WEIGHTS = {
    "narr": 1.0, "neg": 0.3, "three_list": 0.2, "cliche": 5.0,
    "pattern_threshold": 1.0, "burstiness_threshold": 0.3,
    "punct_threshold": 0.03, "ld_threshold": 0.4,
    "ngram_threshold": 0.3, "fw_threshold": 0.5,
    "the_threshold": 0.2, "abstract_threshold": 0.8,
    "commitment_threshold": 0.3,
}
AI_THRESHOLD = 0.35

TAG_TO_KNOWLEDGE = {"3LIST": "THREELIST", "SAME-ENDING": "SAME_ENDING"}


def _knowledge_key(tag: str) -> str:
    return TAG_TO_KNOWLEDGE.get(tag, tag)


class MinerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            envelope = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        payload = envelope.get("payload", envelope)
        text = payload.get("text", "")
        if not text:
            self._respond(400, {"error": "No text provided"})
            return

        regime = payload.get("regime", "short")
        cls = classify(text)
        det = detect(text, source_text=payload.get("source_text"))

        seen_tags = sorted({m["tag"] for m in det.details})
        matches = []
        for m in det.details:
            entry = {
                "tag": m["tag"],
                "line": m["line"],
                "matched_text": m["text"],
            }
            known = glossary([_knowledge_key(m["tag"])], regime)["patterns"]
            k = known.get(_knowledge_key(m["tag"]))
            if k:
                entry["pattern_name"] = k["name"]
                entry["why_slop"] = k["why_slop"]
                entry["suggested_repairs"] = [r["hint"] for r in k["repairs"][:3]]
            matches.append(entry)

        response = {
            "answer": cls["answer"],
            "status": "success",
            "confidence": cls["confidence"],
            "model": cls["model"],
            "matches": matches,
            "guidance": (glossary_compact(det.details) if envelope.get("guidance", "compact") == "compact" else glossary(seen_tags, regime)) if seen_tags else None,
            "breakdown": {
                "pattern_count": sum(1 for _ in det.details),
                "signals": {k: v for k, v in cls["signals"].items()
                            if k in ("narr", "neg", "three_list", "cliche",
                                     "burs", "vocab_div", "fw_ratio", "the_r")},
            },
        }
        self._respond(200, response)

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._respond(200, {
                "status": "ok",
                "miner": "noslop",
                "intent": "AI_TEXT_DETECTION",
                "method": "logreg over v6 patterns + stylometric signals, with fix guidance",
            })
        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), MinerHandler)
    print(f"NoSlop AI Text Detection on :{port}")
    server.serve_forever()
