"""NoSlop Feature Extractor — v6 patterns + LD-Score + n-gram tells + function words.

Combines structural pattern detection with statistical features
to create a feature vector for AI text classification.

Signals:
  1. v6 patterns (NARR, NEG, 3LIST, CLICHE)
  2. LD-Score (letter distribution divergence from English baseline)
  3. N-gram AI tells (known AI phrases)
  4. Function-word ratios (stylometric fingerprint)
  5. Burstiness (sentence length variance)
  6. Vocabulary diversity (type-token ratio)
  7. Punctuation density (em-dash/colon usage)
  8. Fragment ratio (short sentence frequency)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# ─── English letter frequency baseline (from large corpora) ───
ENGLISH_LETTER_FREQ = {
    'a': 0.0817, 'b': 0.0150, 'c': 0.0278, 'd': 0.0425, 'e': 0.1270,
    'f': 0.0223, 'g': 0.0202, 'h': 0.0609, 'i': 0.0697, 'j': 0.0015,
    'k': 0.0077, 'l': 0.0403, 'm': 0.0241, 'n': 0.0675, 'o': 0.0751,
    'p': 0.0193, 'q': 0.0010, 'r': 0.0599, 's': 0.0633, 't': 0.0906,
    'u': 0.0276, 'v': 0.0098, 'w': 0.0236, 'x': 0.0015, 'y': 0.0197,
    'z': 0.0007,
}

# ─── Function words (high-frequency words that carry style signal) ───
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'don', 'should', 'now', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'would', 'could', 'should', 'may', 'might',
    'shall', 'might', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
    'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    'if', 'because', 'although', 'though', 'while', 'unless', 'until',
    'since', 'about', 'against', 'among', 'within', 'without', 'upon',
}

# ─── AI n-gram tells (phrases AI models overuse) ───
# Extended with excess-vocabulary focal words (Kobak et al. 2024/2025,
# Juzek & Ward 2025) and reply-bot rhetoric specimens (slop-lint corpus).
AI_NGRAMS = [
    # Noun phrases
    "tapestry", "landscape of", "realm of", "ecosystem of",
    "paradigm shift", "nuanced perspective", "multifaceted approach",
    "intricate balance", "holistic view", "robust framework",
    "transformative impact", "paramount importance",
    "deep dive", "dive deeper", "shed light",
    "it is worth noting", "it is important to note",
    "plays a crucial role", "pivotal moment",
    "in this article", "in this guide", "in this section",
    "delve into", "navigate the", "embark on",
    "a comprehensive understanding", "a comprehensive guide",
    "leverage the", "harness the", "unlock the",
    "game changer", "game-changer", "level up",
    "the bottom line", "the key takeaway", "to sum up",
    "in conclusion", "ultimately", "at the end of the day",
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "however", "on the other hand",
    "in light of", "in the realm of", "when it comes to",
    "as we delve", "as we explore", "as we navigate",
    "a testament to", "a reflection of",
    "the significance of", "the importance of",
    "stands as a", "serves as a", "acts as a",
    "in today's world", "in today's digital landscape",
    "the power of", "the beauty of", "the art of",
    "whether you're", "whether you are",
    "not only... but also", "from x to y",
]

# Excess-vocabulary focal words (documented LLM overrepresentation)
AI_FOCAL_WORDS = {
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "showcasing", "showcases", "pivotal", "intricate", "intricacies",
    "meticulous", "meticulously", "realm", "realms", "tapestry", "tapestries",
    "commendable", "bolster", "bolstering", "garnered", "unwavering",
    "seamless", "seamlessly", "multifaceted", "nuanced", "holistic",
    "cutting-edge", "groundbreaking", "unparalleled", "ever-evolving",
    "fostering", "elevate", "elevating", "surpass", "surpassing",
    "resonate", "resonates", "aligns", "aligning", "underpins", "underpinning",
}

# Reply-bot / assistant rhetoric openers (structural tells)
REPLY_BOT_PATTERNS = [
    r"(?i)^\s*that'?s (?:the )?distinction",
    r"(?i)\byou'?re right that\b",
    r"(?i)^great question\b",
    r"(?i)\bwhat you'?re describing\b",
    r"(?i)\bit'?s not just .{2,40}? it'?s\b",
    r"(?i)\bthe (?:danger|beauty|genius) (?:of|isn'?t)\b.{0,60}\b(?:is|isn'?t)\b",
    r"(?i)\bisn'?t just\b.{2,50}?\b(?:it'?s|it is)\b",
    r"(?i)\blet'?s unpack\b",
    r"(?i)\bworth noting\b",
]


# ─── v6 Pattern Detection ───

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
    r"(?i)(?:^|(?<=\s))Without\s+[^.,]{3,40},",
    r"(?i)(?:^|(?<=\s))Instead\s+of\s+[^.,]{3,40},",
    r"(?i)(?:^|(?<=\s))What\s+[^,]{3,30}\s+lacks?\s+is",
    r"(?i)\s+not\s+\w[^.,]{2,30}\s+but\s+",
]

THREE_LIST_PATTERNS = [
    r"(?i)[a-z]+(?:\s+\w+){0,3},\s+[a-z]+(?:\s+\w+){0,3},\s+and\s+[a-z]+(?:\s+\w+){0,3}",
    r"(?i)[a-z]+(?:\s+\w+){0,3},\s+[a-z]+(?:\s+\w+){0,3},\s+[a-z]+(?:\s+\w+){0,3}[.,]",
]

CLICHE_WORDS = {
    "power of now", "living your truth", "your authentic self",
    "sacred journey", "sacred path", "divine feminine", "divine masculine",
    "highest self", "step into your power", "step into your light",
    "hold space for", "trust the process", "honor your journey",
    "energetic field", "vibrational frequency",
}


# ─── Signal functions ───

def count_sentences(text: str) -> int:
    return max(1, len([s for s in re.split(r'[.!?]+(?:\s|$)', text) if s.strip()]))


def detect_patterns(text: str) -> dict:
    counts = {"NARR": 0, "NEG": 0, "3LIST": 0, "CLICHE": 0}
    for line in text.split("\n"):
        for p in NARR_PATTERNS:
            counts["NARR"] += len(re.findall(p, line))
        for p in NEG_PATTERNS:
            counts["NEG"] += len(re.findall(p, line))
        for p in THREE_LIST_PATTERNS:
            counts["3LIST"] += len(re.findall(p, line))
        for cliche in CLICHE_WORDS:
            if cliche in line.lower():
                counts["CLICHE"] += 1
    return counts


def sentence_lengths(text: str) -> list[int]:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return [len(s.split()) for s in sentences]


def vocabulary_diversity(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def burstiness(text: str) -> float:
    lengths = sentence_lengths(text)
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(variance) / mean


def avg_sentence_length(text: str) -> float:
    lengths = sentence_lengths(text)
    return sum(lengths) / len(lengths) if lengths else 0.0


def punctuation_density(text: str) -> float:
    dashes = text.count("—") + text.count("–")
    colons = text.count(":")
    words = len(text.split())
    if words == 0:
        return 0.0
    return (dashes + colons) / words


def fragment_ratio(text: str) -> float:
    lengths = sentence_lengths(text)
    if not lengths:
        return 0.0
    short = sum(1 for l in lengths if l < 6)
    return short / len(lengths)


def ld_score(text: str) -> float:
    """Letter Distribution Score — divergence from English baseline.

    AI text: letter frequencies converge toward global average (low divergence).
    Human text: letter frequencies deviate based on domain/style (high divergence).

    Returns: normalized divergence score (0 = matches baseline, 1 = very different).
    Higher = more likely human (domain-specialized).
    Lower = more likely AI (global-averaged).
    """
    text_lower = text.lower()
    letters = [c for c in text_lower if c.isalpha()]
    if len(letters) < 50:
        return 0.5  # not enough data, neutral

    freq = Counter(letters)
    total = len(letters)

    # Jensen-Shannon divergence (smoothed KL divergence)
    divergence = 0.0
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        p = freq.get(letter, 0) / total
        q = ENGLISH_LETTER_FREQ[letter]
        # Smooth to avoid log(0)
        p = max(p, 1e-10)
        q = max(q, 1e-10)
        m = (p + q) / 2
        divergence += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)

    # Normalize to 0-1 range (typical divergence is 0.001-0.05)
    normalized = min(1.0, divergence / 0.05)
    return normalized


def ngram_tell_count(text: str) -> int:
    """Count known AI n-gram tells in text."""
    text_lower = text.lower()
    count = 0
    for ngram in AI_NGRAMS:
        count += text_lower.count(ngram.lower())
    return count


def ngram_tell_density(text: str) -> float:
    """N-gram tells per 100 words."""
    words = len(text.split())
    if words == 0:
        return 0.0
    return (ngram_tell_count(text) / words) * 100


def function_word_ratio(text: str) -> float:
    """Ratio of function words to total words.

    AI text tends to have higher function-word ratio (more formal/structured).
    Human text varies more (casual = fewer, academic = more).
    """
    words = text.lower().split()
    if not words:
        return 0.0
    fw_count = sum(1 for w in words if w in FUNCTION_WORDS)
    return fw_count / len(words)


def function_word_variance(text: str) -> float:
    """Variance of function-word usage across sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
    if len(sentences) < 2:
        return 0.0
    ratios = []
    for s in sentences:
        words = s.lower().split()
        if words:
            fw = sum(1 for w in words if w in FUNCTION_WORDS)
            ratios.append(fw / len(words))
    if not ratios:
        return 0.0
    mean = sum(ratios) / len(ratios)
    variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    return math.sqrt(variance)


def the_rate(text: str) -> float:
    """Rate of sentences starting with 'The'.

    AI text: >40% of sentences start with "The" (narration bias).
    Human text: <25% (varied openings).
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 5]
    if not sentences:
        return 0.0
    the_count = sum(1 for s in sentences if s.startswith("The ") or s.startswith("the "))
    return the_count / len(sentences)


# Common abstract nouns in AI text
ABSTRACT_NOUNS = {
    "approach", "aspect", "assumption", "capacity", "concept", "conclusion",
    "condition", "consequence", "context", "contrast", "contribution",
    "criteria", "debate", "definition", "development", "dimension", "distinction",
    "emphasis", "evidence", "framework", "fundamental", "hypothesis",
    "implication", "importance", "indication", "insight", "interpretation",
    "investigation", "issue", "nature", "notion", "observation", "occurrence",
    "outcome", "perspective", "phenomenon", "position", "possibility",
    "potential", "principle", "process", "progress", "purpose", "reality",
    "relationship", "requirement", "response", "result", "role", "sense",
    "significance", "situation", "solution", "source", "structure", "subject",
    "substance", "system", "technique", "tendency", "theory", "tradition",
    "understanding", "value", "variation", "view", "way",
}

# Concrete nouns that signal human writing
CONCRETE_NOUNS = {
    "body", "bone", "brush", "candle", "clay", "coin", "dust", "earth",
    "fire", "flint", "floor", "furnace", "glass", "grain", "hand", "iron",
    "knife", "lamp", "leaf", "light", "metal", "mirror", "oil", "paper",
    "pen", "rain", "rock", "salt", "sand", "seed", "shadow", "skin",
    "smoke", "snow", "soil", "stone", "thread", "water", "wood",
}


def abstract_noun_density(text: str) -> float:
    """Ratio of abstract nouns to total words.

    AI text: higher density of abstract nouns.
    Human text: more concrete nouns.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    abstract_count = sum(1 for w in words if w in ABSTRACT_NOUNS)
    concrete_count = sum(1 for w in words if w in CONCRETE_NOUNS)
    total = abstract_count + concrete_count
    if total == 0:
        return 0.5  # neutral
    return abstract_count / total


# Voice commitment markers (opinions, stances, personal engagement)
COMMITMENT_MARKERS = [
    r"(?i)\bI think\b", r"(?i)\bI believe\b", r"(?i)\bI feel\b",
    r"(?i)\bin my opinion\b", r"(?i)\bthe way I see it\b",
    r"(?i)\bnever\b", r"(?i)\balways\b", r"(?i)\bmust\b",
    r"(?i)\bshould\b", r"(?i)\bthe truth is\b",
    r"(?i)\bhonestly\b", r"(?i)\bfrankly\b",
    r"(?i)\bthe key is\b", r"(?i)\bwhat matters is\b",
]


def voice_commitment(text: str) -> float:
    """Presence of opinions and stance markers.

    AI text: low commitment (hedging, neutral).
    Human text: high commitment (opinions, stances).
    """
    markers = 0
    for pattern in COMMITMENT_MARKERS:
        markers += len(re.findall(pattern, text))
    words = len(text.split())
    if words == 0:
        return 0.0
    return min(1.0, (markers / words) * 100)


# Staging detection: "X misses the point. Y." (disguised NEG)
STAGING_PATTERNS = [
    r"(?i)[^.!?]+misses?\s+the\s+point",
    r"(?i)[^.!?]+is\s+(?:wrong|incorrect|mistaken)",
    r"(?i)[^.!?]+fails?\s+to\s+(?:understand|recognize|acknowledge)",
    r"(?i)[^.!?]+overlooks?\s+(?:the|a|an)\s+",
    r"(?i)[^.!?]+ignores?\s+(?:the|a|an)\s+",
]


def staging_count(text: str) -> int:
    """Count staging patterns (disguised NEG)."""
    count = 0
    for pattern in STAGING_PATTERNS:
        count += len(re.findall(pattern, text))
    return count


# Lineage chain detection
LINEAGE_PATTERNS = [
    r"(?i)from\s+\w+\s+(?:through|to)\s+\w+\s+(?:through|to)\s+\w+",
    r"(?i)from\s+\w+\s+(?:through|to)\s+\w+\s+and\s+\w+",
]


def lineage_chains(text: str) -> int:
    """Count flat lineage chains."""
    count = 0
    for pattern in LINEAGE_PATTERNS:
        count += len(re.findall(pattern, text))
    return count


# ─── Feature Vector ───

@dataclass
class FeatureVector:
    """Complete feature vector for AI text classification."""
    # v6 patterns
    narr_count: int
    neg_count: int
    three_list_count: int
    cliche_count: int
    total_patterns: int
    pattern_rate: float

    # Statistical
    sentence_count: int
    avg_sentence_len: float
    burstiness: float
    vocab_diversity: float
    punctuation_density: float
    fragment_ratio: float

    # LD-Score
    ld_score: float  # letter distribution divergence

    # N-gram tells
    ngram_tells: int
    ngram_density: float

    # Function words
    function_word_ratio: float
    function_word_variance: float

    # Voice/style signals (from v6/goodprose analysis)
    the_rate: float  # rate of sentences starting with "The"
    abstract_noun_density: float  # abstract vs concrete noun ratio
    voice_commitment: float  # presence of opinions/stance markers
    staging_count: int  # disguised NEG: "X misses the point. Y."
    lineage_chains: int  # "from A through B to C and D"

    # Derived
    is_ai: bool
    confidence: float

    def to_dict(self) -> dict:
        return {
            "narr": self.narr_count,
            "neg": self.neg_count,
            "three_list": self.three_list_count,
            "cliche": self.cliche_count,
            "total_patterns": self.total_patterns,
            "pattern_rate": round(self.pattern_rate, 4),
            "sentences": self.sentence_count,
            "avg_sent_len": round(self.avg_sentence_len, 2),
            "burstiness": round(self.burstiness, 4),
            "vocab_diversity": round(self.vocab_diversity, 4),
            "punct_density": round(self.punctuation_density, 4),
            "fragment_ratio": round(self.fragment_ratio, 4),
            "ld_score": round(self.ld_score, 4),
            "ngram_tells": self.ngram_tells,
            "ngram_density": round(self.ngram_density, 4),
            "fw_ratio": round(self.function_word_ratio, 4),
            "fw_variance": round(self.function_word_variance, 4),
            "the_rate": round(self.the_rate, 4),
            "abstract_nouns": round(self.abstract_noun_density, 4),
            "voice_commitment": round(self.voice_commitment, 4),
            "staging": self.staging_count,
            "lineage": self.lineage_chains,
            "is_ai": self.is_ai,
            "confidence": round(self.confidence, 4),
        }


# ─── Canonical signal set (shared by training and serving) ───

def focal_word_count(text: str) -> int:
    """Excess-vocabulary focal words per 100 words (Kobak et al.)."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,;:!?\"'()") in AI_FOCAL_WORDS)
    return (hits / len(words)) * 100


def em_dash_rate(text: str) -> float:
    """Em-dashes per 100 words — additive-qualifier tell."""
    words = len(text.split())
    if not words:
        return 0.0
    return ((text.count("—") + text.count("--")) / words) * 100


def connector_density(text: str) -> float:
    """Formal connective density per sentence (arXiv 2603.17522 discriminative feature)."""
    sentences = count_sentences(text)
    if not sentences:
        return 0.0
    connectors = ("moreover", "furthermore", "additionally", "consequently",
                  "nevertheless", "nonetheless", "therefore", "thus", "hence",
                  "in contrast", "conversely", "accordingly")
    text_lower = text.lower()
    hits = sum(text_lower.count(c) for c in connectors)
    return hits / sentences


def hedge_count(text: str) -> int:
    """Hedging register markers (LLM assertion-whitening)."""
    patterns = [
        r"(?i)\bit'?s worth noting\b", r"(?i)\bgenerally speaking\b",
        r"(?i)\bin many cases\b", r"(?i)\bwhile this may vary\b",
        r"(?i)\bit is important to note\b", r"(?i)\barguably\b",
        r"(?i)\bin some sense\b", r"(?i)\bto some extent\b",
        r"(?i)\bcould be argued\b", r"(?i)\bit should be noted\b",
    ]
    return sum(len(re.findall(p, text)) for p in patterns)


def reply_bot_openers(text: str) -> int:
    """Validate-then-restate / assistant rhetoric openers."""
    return sum(len(re.findall(p, text)) for p in REPLY_BOT_PATTERNS)


def _pattern_rates(t):
    p = detect_patterns(t)
    n = max(1, count_sentences(t))
    return {k: p[k] / n for k in ("NARR", "NEG", "3LIST", "CLICHE")}


def _pattern_counts(t):
    return detect_patterns(t)


# ─── Competitor-derived signals (slop-cop / dslop imports, 2026-08) ───

def _sent_words(text: str) -> list[list[str]]:
    sents = re.split(r"[.!?\n]+", text)
    return [s.split() for s in sents if len(s.split()) > 0]


def staccato_burst(text: str) -> int:
    """Max run of consecutive sentences with <= 8 words (slop-cop)."""
    runs = run = 0
    for words in _sent_words(text):
        run = run + 1 if len(words) <= 8 else 0
        runs = max(runs, run)
    return max(0, runs - 2)


def anaphora_abuse(text: str) -> int:
    """Max run of consecutive sentences sharing the same first word (slop-cop)."""
    firsts = [w[0].lower() for w in _sent_words(text) if w]
    if len(firsts) < 3:
        return 0
    runs = run = 1
    for a, b in zip(firsts, firsts[1:]):
        run = run + 1 if a == b else 1
        runs = max(runs, run)
    return max(0, runs - 2)


def negation_countdown(text: str) -> int:
    """Max run of consecutive sentences starting with 'Not' (slop-cop)."""
    starts = [1 if words and words[0].lower() == "not" else 0
              for words in _sent_words(text)]
    runs = run = 0
    for s in starts:
        run = run + 1 if s else 0
        runs = max(runs, run)
    return runs


def colon_elaboration(text: str) -> int:
    """Short clause followed by long elaboration via colon (slop-cop shape rule)."""
    return len(re.findall(r"[^.!?\n:]{5,50}:[^:\n]{20,}", text))


def question_then_answer(text: str) -> int:
    """Sentence ending ? immediately followed by a declarative (slop-cop)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sum(1 for a, b in zip(sents, sents[1:])
               if a.endswith("?") and not b.endswith("?") and len(b) <= 200)


def superficial_analysis(text: str) -> int:
    """', highlighting its importance,' trailing-participle frames (slop-cop)."""
    return len(re.findall(
        r",\s+(?:highlighting|underscoring|showcasing|reflecting|cementing|"
        r"embodying|encapsulating)\s+(?:its|the|their|this)\s+"
        r"(?:importance|role|significance|legacy|power|spirit|nature|value)",
        text, re.I))


OPENER_PATTERNS = [
    r"^\s*in\s+an?\s+(?:era|age|world|time)\s+(?:of|where|when)",
    r"^\s*imagine\s+(?:a\s+world|if\s+you|what\s+would|a\s+future)",
    r"^\s*(?:here'?s|here is)\s+the\s+(?:kicker|thing|twist|catch)",
    r"^\s*(?:in\s+conclusion|to\s+conclude|in\s+summary|to\s+sum\s+up|"
    r"in\s+closing|all\s+in\s+all|at\s+the\s+end\s+of\s+the\s+day)",
]


def opener_family_count(text: str) -> int:
    """Sentence-start anchored AI opener families (slop-cop, anchored)."""
    count = 0
    for line in text.split("\n"):
        for sent in re.split(r"(?<=[.!?])\s+", line):
            for p in OPENER_PATTERNS:
                if re.match(p, sent.strip(), re.I):
                    count += 1
    return count


def vague_attribution(text: str) -> int:
    """'experts say', 'studies show' — unattributed authority (slop-cop)."""
    return len(re.findall(
        r"\b(?:experts?|analysts?|observers?|critics?|researchers?)\s+"
        r"(?:argue|say|said|suggest|suggests|believe|note|warn)\b|"
        r"\bstudies\s+show\b|\bresearch\s+(?:shows|suggests)\b|"
        r"\bsources\s+say\b|\bit'?s\s+(?:widely|often)\s+(?:said|believed)\b",
        text, re.I))


def parenthetical_qualifier(text: str) -> int:
    """Long parentheticals and comma-bracketed hedges (slop-cop)."""
    long_parens = len(re.findall(r"\([^)]{20,}\)", text))
    hedges = len(re.findall(
        r",\s+(?:of course|to be fair|admittedly|needless to say|"
        r"it must be said|granted),", text, re.I))
    return long_parens + hedges


def almost_hedge(text: str) -> int:
    return len(re.findall(
        r"\balmost\s+(?:always|never|certainly|exclusively|entirely|"
        r"completely|invariably|universally)\b", text, re.I))


def listicle_magic(text: str) -> int:
    """Markdown/bulleted lists with exactly 3/5/7/10 items (slop-cop)."""
    counts = []
    cur = 0
    for line in text.split("\n"):
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            cur += 1
        elif cur:
            counts.append(cur)
            cur = 0
    if cur:
        counts.append(cur)
    return sum(1 for c in counts if c in (3, 5, 7, 10))


def sentence_len_kurtosis(text: str) -> float:
    """Excess kurtosis of sentence lengths (dslop metric)."""
    lens = [len(s.split()) for s in re.split(r"[.!?]+", text) if s.strip()]
    n = len(lens)
    if n < 3:
        return 0.0
    m = sum(lens) / n
    var = sum((x - m) ** 2 for x in lens) / n
    if var == 0:
        return 0.0
    k4 = sum((x - m) ** 4 for x in lens) / n
    return k4 / (var ** 2) - 3.0


def word_freq_dispersion(text: str) -> float:
    """Top-8 word concentration per 150-word chunk, max over chunks (dslop)."""
    words = [w.lower().strip(".,;:!?\"'()") for w in text.split()
             if w.lower().strip(".,;:!?\"'()")]
    if len(words) < 50:
        return 0.0
    stop = set("the a an and or but of to in on at for with by from as is are "
               "was were be been it its this that these those i you he she "
               "they we not".split())
    best = 0.0
    for i in range(0, len(words) - 49, 150):
        chunk = [w for w in words[i:i + 150] if w not in stop]
        if len(chunk) < 30:
            continue
        top = Counter(chunk).most_common(8)
        conc = sum(c for _, c in top) / len(chunk)
        best = max(best, conc)
    return best


FEATURE_FNS = {
    "narr": lambda t: _pattern_counts(t)["NARR"],
    "neg": lambda t: _pattern_counts(t)["NEG"],
    "three_list": lambda t: _pattern_counts(t)["3LIST"],
    "cliche": lambda t: _pattern_counts(t)["CLICHE"],
    "narr_rate": lambda t: _pattern_rates(t)["NARR"],
    "neg_rate": lambda t: _pattern_rates(t)["NEG"],
    "three_list_rate": lambda t: _pattern_rates(t)["3LIST"],
    "cliche_rate": lambda t: _pattern_rates(t)["CLICHE"],
    "n_sent": count_sentences,
    "avg_sent_len": avg_sentence_length,
    "burs": burstiness,
    "vocab_div": vocabulary_diversity,
    "punct": punctuation_density,
    "frag_ratio": fragment_ratio,
    "ld": ld_score,
    "ngram_dens": ngram_tell_density,
    "focal_words": focal_word_count,
    "em_dash_rate": em_dash_rate,
    "connector_dens": connector_density,
    "hedges": hedge_count,
    "reply_bot": reply_bot_openers,
    "fw_ratio": function_word_ratio,
    "fw_variance": function_word_variance,
    "the_r": the_rate,
    "abs_noun": abstract_noun_density,
    "commitment": voice_commitment,
    "staging": staging_count,
    "lineage": lineage_chains,
    # competitor imports (slop-cop, dslop)
    "staccato": staccato_burst,
    "anaphora": anaphora_abuse,
    "neg_countdown": negation_countdown,
    "colon_elab": colon_elaboration,
    "quest_answ": question_then_answer,
    "superficial": superficial_analysis,
    "openers": opener_family_count,
    "vague_attr": vague_attribution,
    "paren_qual": parenthetical_qualifier,
    "almost_hedge": almost_hedge,
    "listicle3_5_7_10": listicle_magic,
    "sent_kurtosis": sentence_len_kurtosis,
    "word_conc": word_freq_dispersion,
}
FEATURE_NAMES = list(FEATURE_FNS)


def extract(text: str, weights: dict | None = None) -> FeatureVector:
    """Extract feature vector from text using all signals."""
    if weights is None:
        weights = {
            "narr": 1.0, "neg": 1.0, "three_list": 1.0, "cliche": 1.0,
            "pattern_threshold": 0.5,
            "burstiness_threshold": 0.3,
            "punct_threshold": 0.02,
            "ld_threshold": 0.3,
            "ngram_threshold": 0.5,
            "fw_threshold": 0.45,
            "the_threshold": 0.35,
            "abstract_threshold": 0.6,
            "commitment_threshold": 0.1,
        }

    patterns = detect_patterns(text)
    n_sent = count_sentences(text)

    total_patterns = (
        patterns["NARR"] * weights.get("narr", 1.0) +
        patterns["NEG"] * weights.get("neg", 1.0) +
        patterns["3LIST"] * weights.get("three_list", 1.0) +
        patterns["CLICHE"] * weights.get("cliche", 1.0)
    )
    pattern_rate = total_patterns / n_sent if n_sent > 0 else 0

    avg_sent = avg_sentence_length(text)
    burs = burstiness(text)
    vocab = vocabulary_diversity(text)
    punct = punctuation_density(text)
    frag = fragment_ratio(text)
    ld = ld_score(text)
    ngram_count = ngram_tell_count(text)
    ngram_dens = ngram_tell_density(text)
    fw_ratio = function_word_ratio(text)
    fw_var = function_word_variance(text)
    the_r = the_rate(text)
    abs_noun = abstract_noun_density(text)
    commitment = voice_commitment(text)
    staging = staging_count(text)
    lineage = lineage_chains(text)

    # Multi-signal scoring
    score = 0.0

    # Signal 1: v6 patterns (25% weight)
    score += min(pattern_rate / weights.get("pattern_threshold", 0.5), 1.0) * 0.25

    # Signal 2: LD-Score (15% weight) — low LD = AI (matches global baseline)
    ld_threshold = weights.get("ld_threshold", 0.3)
    score += max(0, 1.0 - ld / ld_threshold) * 0.15

    # Signal 3: N-gram tells (10% weight)
    ngram_threshold = weights.get("ngram_threshold", 0.5)
    score += min(ngram_dens / ngram_threshold, 1.0) * 0.10

    # Signal 4: Burstiness (10% weight) — low burstiness = AI
    score += max(0, 1.0 - burs / weights.get("burstiness_threshold", 0.3)) * 0.10

    # Signal 5: "The" rate (10% weight) — high = AI
    the_threshold = weights.get("the_threshold", 0.35)
    score += max(0, min(1.0, the_r / the_threshold)) * 0.10

    # Signal 6: Abstract noun density (10% weight) — high = AI
    abstract_threshold = weights.get("abstract_threshold", 0.6)
    score += max(0, min(1.0, abs_noun / abstract_threshold)) * 0.10

    # Signal 7: Voice commitment (10% weight) — low = AI
    commitment_threshold = weights.get("commitment_threshold", 0.1)
    score += max(0, 1.0 - commitment / commitment_threshold) * 0.10

    # Signal 8: Punctuation density (5% weight) — low = AI
    score += max(0, 1.0 - punct / weights.get("punct_threshold", 0.02)) * 0.05

    # Signal 9: Staging + lineage (5% weight) — presence = AI
    staging_score = min(1.0, (staging + lineage) / 3)
    score += staging_score * 0.05

    confidence = max(0.05, min(0.99, score))

    return FeatureVector(
        narr_count=patterns["NARR"],
        neg_count=patterns["NEG"],
        three_list_count=patterns["3LIST"],
        cliche_count=patterns["CLICHE"],
        total_patterns=int(total_patterns),
        pattern_rate=pattern_rate,
        sentence_count=n_sent,
        avg_sentence_len=avg_sent,
        burstiness=burs,
        vocab_diversity=vocab,
        punctuation_density=punct,
        fragment_ratio=frag,
        ld_score=ld,
        ngram_tells=ngram_count,
        ngram_density=ngram_dens,
        function_word_ratio=fw_ratio,
        function_word_variance=fw_var,
        the_rate=the_r,
        abstract_noun_density=abs_noun,
        voice_commitment=commitment,
        staging_count=staging,
        lineage_chains=lineage,
        is_ai=confidence > 0.5,
        confidence=confidence,
    )
