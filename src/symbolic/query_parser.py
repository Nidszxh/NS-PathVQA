"""Rule-based question parser that classifies PathVQA questions into query types.

Uses regex patterns in priority order:
  1. Attribute (color/shape/size/type)
  2. Count (how many)
  3. Yes/no (is/are/does/do/can/has/have/was/were)
  4. Identity (what/which/identify)
  5. Location (where/in which organ/system)

Returns a structured Query dataclass with qtype, target, and optional attribute.

The priority ordering ensures that questions matching multiple patterns
are classified by the most specific type first. For example, "what color is the lesion?"
matches both attribute and identity, but attribute is checked first.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Query:
    """Structured representation of a parsed question.

    Attributes:
        qtype: Type of question (identity, location, yes_no, attribute, count)
        target: Extracted target word/phrase from the question (if any)
        attribute: For attribute questions, the attribute type (color/shape/size/type)
        value: For attribute questions, the predicted value (populated after execution)
    """
    qtype: str
    target: str = ""
    attribute: str = ""
    value: str = ""


IDENTITY_PATTERNS = [
    r"what (?:is|are|does) (?:this|the|that|there|being|it) (?:show|shown|seen?|display|present|demonstrat)",
    r"what (?:is )?the (?:diagnosis|finding|lesion|abnormality)",
    r"what (?:region|organ|structure|tissue|system) (?:is|are)",
    r"what (?:do|does) .*? (?:show|demonstrat|depict|illustrat)",
    r"identify (?:the |this )?(\w+)",
]

LOCATION_PATTERNS = [
    r"^where ",
    r"from where",
    r"what (?:is the )?(?:location|site|origin|source)",
    r"which (?:organ|system|region|part|area|structure)",
    r"what (?:organ|system|region|part|area|structure)",
]

ATTRIBUTE_PATTERNS = {
    "color": r"what colo[u]?r",
    "size": r"what (?:size|is the size)",
    "shape": r"what shape",
    "type": r"what (?:type|kind|variety)",
}


def _extract_target(question: str, answer_vocab: List[str]) -> str:
    """Find the most specific word/phrase from the question that appears in the answer vocabulary."""
    q = question.lower()
    words = set(re.findall(r"\b[a-z]+\b", q))
    for word in sorted(words, key=len, reverse=True):
        if word in answer_vocab:
            return word
    for phrase_len in range(4, 0, -1):
        for i in range(len(q.split()) - phrase_len + 1):
            phrase = " ".join(q.split()[i:i+phrase_len])
            if phrase in answer_vocab:
                return phrase
    return ""


def parse_question(question: str, answer_vocab: Optional[List[str]] = None) -> Query:
    """Classify a question string into a structured Query.

    Pattern priority: attribute > count > yes_no > identity > location > default identity.

    Args:
        question: Raw question string
        answer_vocab: List of known answer strings (for target extraction)

    Returns:
        Query with classified qtype and optionally extracted target/attribute
    """
    q = question.lower().strip()
    answer_vocab = answer_vocab or []
    target = _extract_target(q, answer_vocab) if answer_vocab else ""

    # Attribute questions (checked first: "what color", "what shape", etc.)
    for attr, pattern in ATTRIBUTE_PATTERNS.items():
        if re.search(pattern, q):
            return Query(qtype="attribute", target=target, attribute=attr)

    # Count questions ("how many...")
    if re.search(r"^how many ", q):
        return Query(qtype="count", target=target)

    # Yes/no questions ("is/are/does this...")
    if re.search(r"^(?:is|are|does|do|can|has|have|was|were) ", q):
        return Query(qtype="yes_no", target=target)

    # Identity questions ("what...", "identify...")
    for pattern in IDENTITY_PATTERNS:
        if re.search(pattern, q):
            return Query(qtype="identity", target=target)

    # Location questions ("where...", "which organ...", "what region...")
    for pattern in LOCATION_PATTERNS:
        if re.search(pattern, q):
            return Query(qtype="location", target=target)

    # Default fallback: treat as identity question
    return Query(qtype="identity", target=target)


if __name__ == "__main__":
    test_questions = [
        "what color is the lesion?",
        "how many nuclei are present?",
        "is this a malignant tumor?",
        "what is shown in this image?",
        "where is the abnormality located?",
        "what shape are the cells?",
    ]
    dummy_vocab = ["lesion", "malignant", "tumor", "nuclei", "cells"]
    for q in test_questions:
        result = parse_question(q, dummy_vocab)
        print(f"  {q:45s} → qtype={result.qtype:10s} target={result.target:15s} attr={result.attribute}")
    print("QueryParser test passed!")
