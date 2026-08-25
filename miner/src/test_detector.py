#!/usr/bin/env python3
"""Tests for NoSlop AI Text Detection engine."""
import sys
sys.path.insert(0, ".")
from detector import detect

def test_heavy_ai_slop():
    text = """The paper opens with alchemy, and the choice is deliberate.
    Not conceptual analysis, but participatory knowledge.
    Gods, angels, and daimons. The power of now is something we all need.
    This section examines the role of imagination.
    We now turn to the question of consciousness."""
    r = detect(text)
    assert r.is_ai == True, f"Expected AI, got {r.is_ai}"
    assert r.confidence > 0.5, f"Expected conf>0.5, got {r.confidence}"
    assert r.pattern_count >= 5, f"Expected >=5 patterns, got {r.pattern_count}"
    assert "NARR" in r.patterns_found, f"Expected NARR, got {r.patterns_found}"
    print(f"PASS heavy_slop: {r.pattern_count} patterns, conf={r.confidence}")

def test_clean_prose():
    text = """The Burning Bush makes this concrete. A brushwood fire to the senses
    — a theophany to the imagination. Same phenomenon, different organ of perception.
    The active Imagination reveals what is already there, what lies beyond the
    physical eye's range. Integration costs something. Learning to steer is the
    entire discipline."""
    r = detect(text)
    assert r.pattern_count <= 2, f"Expected <=2 patterns for clean text, got {r.pattern_count}"
    print(f"PASS clean_prose: {r.pattern_count} patterns, conf={r.confidence}")

def test_three_list_detection():
    text = "The gods, angels, and daimons appeared in the vision."
    r = detect(text)
    assert "3LIST" in r.patterns_found, f"Expected 3LIST in {r.patterns_found}"
    print(f"PASS three_list: found 3LIST")

def test_narr_detection():
    text = "The paper opens with a distinction between two modes of knowing."
    r = detect(text)
    assert "NARR" in r.patterns_found, f"Expected NARR in {r.patterns_found}"
    print(f"PASS narr: found NARR")

def test_cliche_detection():
    text = "We must step into our power and live our truth with the power of now."
    r = detect(text)
    assert "CLICHE" in r.patterns_found, f"Expected CLICHE in {r.patterns_found}"
    print(f"PASS cliche: found CLICHE")

def test_human_text_low_score():
    text = """Barzakh. The word means barrier, isthmus, something that separates
    two things without becoming either of them. Ibn Arabi turns this ordinary
    Arabic word into the name for divine Imagination itself. The pun is the whole
    theology compressed into a single word."""
    r = detect(text)
    assert r.confidence < 0.8, f"Expected low confidence for human text, got {r.confidence}"
    print(f"PASS human_text: conf={r.confidence}")

if __name__ == "__main__":
    test_heavy_ai_slop()
    test_clean_prose()
    test_three_list_detection()
    test_narr_detection()
    test_cliche_detection()
    test_human_text_low_score()
    print("\nAll tests passed.")
