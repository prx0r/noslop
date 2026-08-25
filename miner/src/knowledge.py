"""NoSlop pattern knowledge base for agent-facing fix guidance.

Distilled from the writing corpus:
  - noslop/algorithms/v6algorithm.md (zero-tolerance regime)
  - noslop/algorithms/v7algorithm.md + longform-algorithm.md (purpose-tested regime)
  - noslop/anti-slop/{stopslop,antislop,antislopguide}.md (repair tools)
  - workengestation .../writing/essay/{goodprose,rumiengine,slopnotes,slopreview}.md

Master rule (v7): a pattern is not a violation when you choose it.
It is a violation when it chooses you.
"""
from __future__ import annotations

REGIMES = {
    "short": {
        "name": "zero-tolerance",
        "applies_to": "commentary-block format: AI blocks of 20-40 words interleaved with source blocks",
        "rule": "Every tagged pattern must be eliminated. Counts > 0 fail.",
    },
    "long": {
        "name": "purpose-tested",
        "applies_to": "continuous argument, 3000-8000 words, academic or essay register",
        "rule": "A pattern is a violation only when it is a crutch. Chosen patterns "
                "with orienting function are permitted within budgets.",
    },
}

PATTERN_KNOWLEDGE = {
    "NARR": {
        "name": "Narratitivitis",
        "definition": "The sentence narrates what the text or author does instead of stating the idea directly.",
        "why_slop": "The reader came for content, not a tour guide. Narration puts a layer "
                    "of glass between reader and idea, and signals summary rather than claim.",
        "disguises": [
            "self-narration: 'The paper opens with...'",
            "author-narration: 'X introduces / develops / argues...'",
            "section-narration: 'This section examines...'",
            "collective-narration: 'We now turn to...'",
            "preview-narration: 'What follows is an analysis of...'",
        ],
        "repairs": [
            {"tool": "delete_and_state",
             "hint": "Delete the narration clause and state the idea directly. If nothing "
                     "remains, there was no claim.",
             "before": "The paper opens with alchemy, and the choice is deliberate.",
             "after": "Alchemy is the controlling metaphor for everything that follows."},
            {"tool": "lead_with_concept",
             "hint": "Vary the subject: lead with the concept, not the author.",
             "before": "Ibn Arabi develops the idea of barzakh...",
             "after": "Barzakh is the whole theology compressed into a single word."},
        ],
        "budget_short": "Zero tolerance. One exception per essay: the first AI block may "
                        "position the author ('Plotinus begins with inward turning').",
        "budget_long": "Max one orienting narration per 500 words. Test: is it doing "
                       "orienting work (telling the reader where they are) or padding?",
    },
    "NEG": {
        "name": "Negat-assert-itis",
        "definition": "The sentence clears ground by negating a wrong view before asserting "
                      "the right one: 'Not X, but Y.'",
        "why_slop": "The negation is scaffolding, not meaning. Readers learn to skip the first "
                    "half of every sentence and the prose feels defensive rather than confident.",
        "disguises": [
            "fragment: 'Not X. Y.'",
            "clause: 'X, not Y'",
            "imperative: 'Forget X. Y.'",
            "condition: 'Without X, Y.'",
            "substitution: 'Instead of X, Y'",
            "lack: 'What X lacks is Y'",
            "insufficiency: 'X alone cannot Y'",
            "comparison: 'The difference between X and Y...'",
            "staging: wrong idea in sentence 1, correction in sentence 2",
        ],
        "repairs": [
            {"tool": "flat_assertion",
             "hint": "State what it IS. No negation, no implied negation.",
             "before": "Not projections of the psyche, but presences encountered in a realm.",
             "after": "Presences encountered in a realm."},
            {"tool": "forget_opener",
             "hint": "Dismiss without the negate-then-assert rhythm.",
             "before": "The imagination is not fantasy or daydreaming.",
             "after": "Forget the imagination you think you know. Corbin means something else entirely."},
            {"tool": "question_answer",
             "hint": "Let the question do the negating implicitly.",
             "before": "Not the senses. Not the intellect. A third faculty.",
             "after": "What organ sees it? The active Imagination."},
            {"tool": "fragment_momentum",
             "hint": "Noun fragments carrying weight, verbless.",
             "after": "Theophanic visions. Dreams. Meditative states."},
            {"tool": "concrete_image",
             "hint": "Show the consequence instead of negating.",
             "before": "X is not academic; it determines how you live.",
             "after": "You either study the imaginal world or you live in it."},
            {"tool": "positive_after_negation",
             "hint": "If the negation is essential, keep it but put the positive assertion "
                     "immediately after, joined by an em-dash.",
             "after": "The eye is not a passive receiver — it participates in what it sees."},
        ],
        "semantic_inversion_warning": (
            "When removing a negation, NEVER flip the sentence into its opposite "
            "('not passive' -> 'passive'). Restate the true positive claim with equal force. "
            "This bug occurred 12 times in documented runs."
        ),
        "budget_short": "Max 2 per essay; ideally 1 per 800-1000 words. Zero tolerance on "
                        "stacked forms. If any form appeared in the last 3 blocks, the next gets none.",
        "budget_long": "Permitted only with all three: named target (not straw man), evidence "
                       "it is wrong, positive claim after. Missing any one -> rewrite.",
        "exception": "Apophatic subjects (via negativa): if the subject can ONLY be approached "
                     "through negation, negation may be chosen structure, not crutch.",
    },
    "THREELIST": {
        "name": "Three-item list-itis",
        "definition": "Three grammatically parallel items in sequence, evenly weighted.",
        "why_slop": "The reader's brain recognizes the cadence before processing content. Items "
                    "blur together because none carries weight — rhythm without meaning.",
        "disguises": [
            "4 evenly-spaced items (same cadence — adding a fourth does NOT dodge the rule)",
            "flat lineage chain: 'from Plotinus through Iamblichus to Jung and Hillman'",
        ],
        "repairs": [
            {"tool": "develop_one",
             "hint": "Pick one item and give it the whole sentence; let the others fall away. "
                     "Weight, not count.",
             "before": "a journey to the underworld, a fight with the dragon, a nigredo",
             "after": "rot before it can fruit"},
            {"tool": "add_gravity",
             "hint": "Differentiate otherwise-flat parallels by giving items individual weight.",
             "before": "gods, angels, and daimons",
             "after": "gods, angels, daimons — each with its own gravity"},
            {"tool": "absorb_into_structure",
             "hint": "Absorb the items into sentence structure, e.g. contrast two instead of listing three.",
             "after": "One directs it downward toward matter, the other upward toward archetypes."},
        ],
        "accumulation_test": "A longer list is legitimate only if items are unequal in length/weight "
                             "and the list builds toward a claim. Collapse test: if any item could be "
                             "removed without loss, it is enumeration — collapse it.",
        "budget_short": "Zero tolerance, including 4-item and lineage-chain disguises.",
        "budget_long": "Accumulation permitted; enumeration never.",
    },
    "PARA": {
        "name": "Paraphrase-itis",
        "definition": "A passage restates what the source just said in different words, same claims.",
        "why_slop": "The reader already read the source. Restating wastes attention; every passage "
                    "must add what the source doesn't say.",
        "disguises": [
            "framing blocks that only set up what the reader already has",
            "copying the source's headline phrase",
            "analysis that could be deleted with no loss (framing work, not depth work)",
        ],
        "repairs": [
            {"tool": "delete_test",
             "hint": "If removing the passage loses nothing, delete it."},
            {"tool": "add_mechanism_image",
             "hint": "Add one thing the source lacks: a mechanism, an image, a consequence, "
                     "a connection, or a stance.",
             "before": "Ta'wil refers to restoring the true meaning of a text through transmuting the world into symbols.",
             "after": "Ta'wil trains the soul to pare through surfaces, each layer a skin of habitual seeing stripped back."},
        ],
        "ratio_rule": "For every line of quote, at least two lines of analysis that add something new. "
                      "If analysis only restates, cut it and let the quote stand alone. "
                      "A quote without analysis is a museum display.",
        "budget_short": "Zero removable passages tolerated.",
        "budget_long": "Permitted as setup only if it enables analysis that couldn't exist without it.",
    },
    "FLAT": {
        "name": "Museum-guide neutrality",
        "definition": "Uniform sentence length, no fragments, no rhythmic variety; neutral summarizer stance.",
        "why_slop": "Uniform length produces a drone. Prose needs pulse: short beats for emphasis, "
                    "long arcs for complexity, fragments for landing. Predictability is death — read "
                    "aloud; if you can predict the next sentence's shape, it's dead.",
        "disguises": [
            "every sentence 20-35 words",
            "no sentence under 8 words anywhere in a paragraph",
            "neutral summarizer voice (no stance: amused, unsettled, awed, skeptical)",
        ],
        "repairs": [
            {"tool": "rhythm_surgery",
             "hint": "Add one fragment under 8 words, one sentence over 45 words, vary openings. "
                     "Target roughly 20% short / 60% medium / 20% long; max/min length ratio >= 4x.",
             "after": "Integration costs something."},
            {"tool": "em_dash_pivot",
             "hint": "Use [assertion] — [what it means or inverts it].",
             "after": "Without it, brushwood and wind. With it, the Burning Bush."},
            {"tool": "commit_stance",
             "hint": "Sound like someone who half-believes this, or is unsettled by it, or is quietly "
                     "amused by it — never a neutral summarizer."},
            {"tool": "hedge_control",
             "hint": "Max one hedge word ('seems', 'arguably') per paragraph. Hedge in its own short "
                     "sentence, never buried inside the claim.",
             "after": "However uncomfortable that is to sit with."},
        ],
        "budget_short": "Any block without rhythm variety fails. Em-dashes are prescribed texture here, "
                        "not abuse.",
        "budget_long": "Academic register allows 30-50 word sentences; the test is predictability, "
                       "not absolute length.",
    },
    "CLICHE": {
        "name": "Cliche infiltration",
        "definition": "Self-help language dropped into context without irony.",
        "why_slop": "Modern therapeutic idioms break the historical/intellectual register and make "
                    "prose feel like a TED talk in costume.",
        "disguises": ["'the power of now'", "'living your truth'", "'your authentic self'",
                      "'hold space for'", "'step into your power'", "'trust the process'",
                      "'sacred journey'", "'divine feminine'"],
        "repairs": [
            {"tool": "delete", "hint": "Delete it outright."},
            {"tool": "register_from_subject",
             "hint": "Replace with vocabulary drawn from the subject itself — the register should "
                     "reinforce what the material already does."},
        ],
        "budget_short": "Zero. Any count fails.",
        "budget_long": "Zero.",
    },
    "SAME_ENDING": {
        "name": "The dead ending",
        "definition": "The closing echoes the last source passage's words, structure, or phrasing.",
        "why_slop": "The essay ends on someone else's voice. The reader feels a fade-out instead of "
                    "a landing. Summing up is dead; a good ending opens outward.",
        "disguises": [
            "summing up ('In conclusion...', 'Ultimately...')",
            "tagline close",
            "sharing key content words with the source's final passage",
        ],
        "repairs": [
            {"tool": "circle_back_transformed",
             "hint": "Return to your opening image transformed. End on an image, question, or opening "
                     "— not a conclusion. Share zero key words with the source's ending.",
             "after": "A home scored into the fabric of things before the soul arrived to find it."},
            {"tool": "rewrite_ending_first",
             "hint": "Rewrite the ending FIRST, not last — three separate times, cold, before judging it."},
            {"tool": "silent_ending",
             "hint": "If the last passage already carries the payload, stop. Ending silently is allowed."},
        ],
        "budget_short": "Any keyword overlap with the source ending fails.",
        "budget_long": "Final paragraph must not summarize, should shift register slightly "
                       "(analysis -> implication), and end on an image or opening.",
    },
}

SEQUENCING_RULES = [
    "Fix the ending FIRST, then work forward.",
    "One sentence at a time: change it, re-read the surrounding 2-3 sentences, check the fix "
    "didn't introduce a new pattern; undo and try a different approach if it did.",
    "After fixing any NEG: verify you did NOT invert the meaning (see semantic_inversion_warning).",
    "Rotate repair tools: never resolve two consecutive hits the same way.",
    "If more than 5 patterns total: don't spot-fix — rewrite the worst-hit sentences entirely.",
    "Re-run detection after repairs; iterate until clean.",
    "Overcorrection check: a rule-correct fix that drains the drama is still a failure — escalate "
    "to a stronger tool rather than accepting dead prose. Restore one concrete noun, vivid verb, "
    "or unexpected modifier where a fix flattened life.",
]

POSITIVE_TARGETS = [
    "Every abstract noun physically tethered: anchored to something you can photograph within two sentences.",
    "At least one genuinely alive line per passage — a sentence with stance, stakes, or surprise.",
    "One unexpected concrete noun placed at a structural pivot point (where the sentence turns).",
    "Register comes from the subject itself, not from generic literary decoration.",
    "Essence test: could someone who knows nothing here feel something, not just understand something?",
]


def glossary(tags: list[str], regime: str = "short") -> dict:
    """Build the agent-facing knowledge payload for the detected tags."""
    reg = REGIMES.get(regime, REGIMES["short"])
    budget_key = "budget_short" if regime != "long" else "budget_long"
    known = {}
    for tag in tags:
        entry = PATTERN_KNOWLEDGE.get(tag)
        if entry is None:
            continue
        slim = {k: entry[k] for k in ("name", "definition", "why_slop", "disguises", "repairs")
                if k in entry}
        slim[budget_key] = entry.get(budget_key)
        for extra in ("semantic_inversion_warning", "exception", "accumulation_test", "ratio_rule"):
            if extra in entry:
                slim[extra] = entry[extra]
        known[tag] = slim
    return {
        "regime": reg,
        "patterns": known,
        "sequencing_rules": SEQUENCING_RULES,
        "positive_targets": POSITIVE_TARGETS,
    }


def glossary_compact(matches: list[dict]) -> dict:
    """Highlight-shaped guidance: what each tag is and why it reads as slop.

    Lab finding (experiments/LABS.md Stage A): this payload beats the full
    repair-script payload on reduction AND retention at ~1/7 the size.
    """
    by_tag = {}
    order = []
    for m in matches:
        k = TAG_TO_KEY.get(m["tag"], m["tag"])
        if k not in by_tag:
            entry = PATTERN_KNOWLEDGE.get(k)
            if entry:
                by_tag[k] = {"name": entry["name"], "why_slop": entry["why_slop"],
                             "count": 0}
                order.append(k)
        if k in by_tag:
            by_tag[k]["count"] += 1
    return {"patterns": {k: by_tag[k] for k in order}}


TAG_TO_KEY = {"3LIST": "THREELIST", "SAME-ENDING": "SAME_ENDING"}
