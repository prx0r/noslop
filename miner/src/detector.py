#!/usr/bin/env python3
"""NoSlop AI Text Detection Engine — v6 pattern catalog."""
import re, math
from dataclasses import dataclass, field
from typing import Optional

NARR_PATTERNS = [
    r"(?i)(?:the\s+)?(?:paper|article|essay|text|chapter|section|author)\s+opens?\s+with",
    r"(?i)(?:the\s+)?(?:paper|article|essay|text|chapter|section)\s+(?:introduces?|presents?|examines?|explores?|discusses?|considers?|turns?\s+to|closes?\s+with|concludes?\s+with)",
    r"(?i)(?:this|the)\s+section\s+(?:examines?|explores?|discusses?|considers?|analyzes?|investigates?)",
    r"(?i)(?:we|you)\s+(?:now\s+)?(?:turn\s+to|move\s+to)",
    r"(?i)(?:what\s+follows?\s+is|here\s+is)\s+(?:an?\s+)?(?:analysis|examination|discussion|overview|summary)",
    r"(?i)(?:he|she|they)\s+(?:now\s+)?(?:turns?\s+to|moves?\s+to|introduces?|presents?|examines?|develops?|distinguishes?|gives?|places?|describes?|traces?|closes?|concludes?|argues?|notes?|highlights?|emphasizes?|focuses?|offers?|provides?|suggests?|proposes?|claims?|asserts?|contends?|maintains?|posits?)",
    r"(?i)(?:the\s+)?(?:key|main|central|crucial|important)\s+(?:claim|point|idea|argument|insight|observation)\s+(?:is|here\s+is|:)",
]

NEG_PATTERNS = [
    r"(?i)(?:^|(?<=\s))Not\s+[^.,;]{3,40}[,.]?\s+(?:but\s+)?(?:[A-Z])",
    r"(?i),\s+not\s+\w",
    r"(?i)(?:^|(?<=\s))Forget\s+\w",
    r"(?i)(?:^|(?<=\s))Without\s+[^.,]{3,40},",
    r"(?i)(?:^|(?<=\s))Instead\s+of\s+[^.,]{3,40},",
    r"(?i)(?:^|(?<=\s))What\s+[^,]{3,30}\s+lacks?\s+is",
    r"(?i)[^.]{3,30}\s+alone\s+(?:cannot|can't|could\s+not|won't|doesn't|does\s+not)",
    r"(?i)(?:^|(?<=\s))The\s+difference\s+between",
    r"(?i)\s+not\s+\w[^.,]{2,30}\s+but\s+",
]

THREE_LIST_PATTERNS = [
    r"(?i)[a-z]+(?:\s+\w+){0,3},\s+[a-z]+(?:\s+\w+){0,3},\s+and\s+[a-z]+(?:\s+\w+){0,3}",
    r"(?i)[a-z]+(?:\s+\w+){0,3},\s+[a-z]+(?:\s+\w+){0,3},\s+[a-z]+(?:\s+\w+){0,3}[.,]",
]

CLICHE_PATTERNS = [
    r"(?i)the\s+power\s+of\s+now",
    r"(?i)living\s+your\s+truth",
    r"(?i)your\s+authentic\s+self",
    r"(?i)sacred\s+(?:journey|path|space)",
    r"(?i)divine\s+(?:feminine|masculine|energy)",
    r"(?i)highest\s+(?:self|potential)",
    r"(?i)align(?:ing|ed)?\s+with\s+(?:your|the)\s+(?:purpose|energy|truth)",
    r"(?i)(?:soul|spirit)\s+(?:calling|purpose|mate)",
    r"(?i)step\s+into\s+(?:your|our)\s+(?:power|light|truth)",
    r"(?i)hold\s+space\s+for",
    r"(?i)trust\s+the\s+process",
    r"(?i)(?:energetic|vibrational)\s+(?:field|frequency|alignment)",
]

@dataclass
class PatternMatch:
    tag: str
    matched_text: str
    line: int

@dataclass
class DetectionResult:
    is_ai: bool
    confidence: float
    patterns_found: dict = field(default_factory=dict)
    details: list = field(default_factory=list)
    pattern_count: int = 0
    pattern_rate: float = 0.0

def count_sentences(text):
    return max(1, len([s for s in re.split(r'[.!?]+(?:\s|$)', text) if s.strip()]))

def detect_narr(text):
    matches = []
    for i, line in enumerate(text.split(chr(10)), 1):
        for p in NARR_PATTERNS:
            for m in re.finditer(p, line):
                matches.append(PatternMatch("NARR", m.group(), i))
    return matches

def detect_neg(text):
    matches = []
    for i, line in enumerate(text.split(chr(10)), 1):
        for p in NEG_PATTERNS:
            for m in re.finditer(p, line):
                matches.append(PatternMatch("NEG", m.group(), i))
    return matches

def detect_three_list(text):
    matches = []
    for i, line in enumerate(text.split(chr(10)), 1):
        for p in THREE_LIST_PATTERNS:
            for m in re.finditer(p, line):
                matches.append(PatternMatch("3LIST", m.group(), i))
    return matches

def detect_flat(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+(?:\s|$)', text) if s.strip()]
    if len(sentences) < 3:
        return []
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    uniform = sum(1 for l in lengths if abs(l - avg) <= 5)
    if uniform / len(lengths) > 0.7 and avg > 15:
        return [PatternMatch("FLAT", f"avg {avg:.0f} words/sentence", 0)]
    return []

def detect_cliche(text):
    matches = []
    for i, line in enumerate(text.split(chr(10)), 1):
        for p in CLICHE_PATTERNS:
            for m in re.finditer(p, line):
                matches.append(PatternMatch("CLICHE", m.group(), i))
    return matches

def calculate_confidence(total_patterns, total_sentences):
    if total_sentences == 0:
        return 0.0
    rate = total_patterns / total_sentences
    confidence = 1 / (1 + math.exp(-20 * (rate - 0.05)))
    return max(0.05, min(0.99, confidence))

def _dedupe(matches):
    """Drop matches fully contained in a longer match of the same tag and line."""
    kept = []
    for m in sorted(matches, key=lambda x: -len(x.matched_text)):
        if not any(m.tag == k.tag and m.line == k.line and m.matched_text in k.matched_text
                   for k in kept):
            kept.append(m)
    return kept


def detect(text, source_text=None):
    all_matches = []
    all_matches.extend(detect_narr(text))
    all_matches.extend(detect_neg(text))
    all_matches.extend(detect_three_list(text))
    all_matches.extend(detect_flat(text))
    all_matches.extend(detect_cliche(text))
    all_matches = _dedupe(all_matches)

    pattern_counts = {}
    for m in all_matches:
        pattern_counts[m.tag] = pattern_counts.get(m.tag, 0) + 1

    total_sentences = count_sentences(text)
    total_patterns = len(all_matches)
    pattern_rate = total_patterns / total_sentences if total_sentences > 0 else 0
    confidence = calculate_confidence(total_patterns, total_sentences)

    return DetectionResult(
        is_ai=confidence > 0.5,
        confidence=round(confidence, 4),
        patterns_found=pattern_counts,
        details=[{"tag": m.tag, "text": m.matched_text, "line": m.line} for m in all_matches],
        pattern_count=total_patterns,
        pattern_rate=round(pattern_rate, 4),
    )

if __name__ == "__main__":
    import sys, json
    text = sys.stdin.read()
    result = detect(text)
    print(json.dumps({
        "answer": 1 if result.is_ai else 0,
        "status": "success",
        "confidence": result.confidence,
        "patterns": result.patterns_found,
        "pattern_count": result.pattern_count,
        "pattern_rate": result.pattern_rate,
    }, indent=2))
