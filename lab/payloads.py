"""Payload variants for miner->agent guidance, forming an information ladder:

    none < counts < locate < highlight < full

Each level strictly adds information. The ablation asks how much a capable
LLM actually needs. Philosophy: do not over-guide; assume agent context.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miner.src.detector import detect
from miner.src.knowledge import glossary

VARIANTS = ["none", "counts", "locate", "highlight", "full"]


def matches_for(text):
    return detect(text).details


def build_diagnostic(variant: str, text: str) -> str:
    """Diagnostic block injected into the rewrite prompt. '' for 'none'."""
    ms = matches_for(text)
    if variant == "none":
        return ""
    if not ms:
        return "No slop patterns detected."

    if variant == "counts":
        agg = {}
        for m in ms:
            agg[m["tag"]] = agg.get(m["tag"], 0) + 1
        return ("Pattern detector found these AI-slop pattern counts in your text:\n"
                + json.dumps(agg))

    if variant == "locate":
        lines = [f"[{m['tag']}] \"{m['text']}\"" for m in ms[:30]]
        return ("The detector flagged these passages:\n" + "\n".join(lines))

    if variant == "highlight":
        lines = []
        seen_why = set()
        for m in ms[:30]:
            k = glossary([m["tag"]], "long")["patterns"].get(m["tag"], {})
            why = k.get("why_slop", "")
            note = f" ({why})" if m["tag"] not in seen_why and why else ""
            seen_why.add(m["tag"])
            lines.append(f"[{m['tag']}] \"{m['text']}\"{note}")
        return ("The detector flagged these passages as AI-slop patterns "
                "(tag, quote, and why it reads as machine-generated):\n"
                + "\n".join(lines))

    if variant == "full":
        entries = []
        for m in ms[:30]:
            k = glossary([m["tag"]], "long")["patterns"].get(m["tag"], {})
            entries.append({
                "tag": m["tag"],
                "matched_text": m["text"],
                "why_slop": k.get("why_slop"),
                "repairs": [r["hint"] for r in k.get("repairs", [])][:2],
            })
        g = glossary(sorted({m["tag"] for m in ms}), regime="long")
        budgets = {t: e.get("budget_long") for t, e in g["patterns"].items()}
        rules = g["sequencing_rules"]
        return ("Full diagnostic report from the pattern detector:\n"
                + json.dumps({"matches": entries, "budgets": budgets,
                              "repair_rules": rules}, indent=1))

    raise ValueError(f"unknown variant {variant}")


REWRITE_INSTRUCTION_NONE = (
    "Revise the essay to remove AI-slop writing patterns "
    "(narration tics, negation scaffolds, three-item lists, cliches, "
    "uniform rhythm). Preserve meaning, voice, and length.")

REWRITE_INSTRUCTION = (
    "Revise the essay to address these findings. "
    "Preserve meaning, voice, and length.")


def build_full_rewrite_prompt(variant: str, text: str) -> str:
    diag = build_diagnostic(variant, text)
    instr = REWRITE_INSTRUCTION_NONE if variant == "none" else REWRITE_INSTRUCTION
    parts = [f"Here is an essay:\n\n{text}\n"]
    if diag:
        parts.append(f"\nDIAGNOSTIC FINDINGS:\n{diag}\n")
    parts.append(f"\n{instr}\nOutput ONLY the revised essay.")
    return "\n".join(parts)


def build_patch_prompt(variant: str, flagged: list[dict]) -> str:
    """flagged: [{id, sentence, tag}] -> one JSON call returning replacements."""
    items = []
    seen_why = set()
    for f in flagged:
        item = {"id": f["id"], "sentence": f["sentence"], "pattern_tag": f["tag"]}
        if variant in ("highlight", "full"):
            k = glossary([f["tag"]], "long")["patterns"].get(f["tag"], {})
            if f["tag"] not in seen_why and k.get("why_slop"):
                item["why_flagged"] = k["why_slop"]
            if variant == "full" and k.get("repairs"):
                item["suggested_repairs"] = [r["hint"] for r in k["repairs"]][:2]
            seen_why.add(f["tag"])
        items.append(item)
    return ("Each sentence below was flagged by an AI-slop pattern detector. "
            "Return ONLY a JSON object mapping each id to a replacement sentence "
            "that removes the pattern while fitting the surrounding prose and "
            "preserving meaning.\n\n" + json.dumps(items, indent=1))
