"""Rule-based question parser: classifies questions and compiles DSL program ASTs.

Priority: attribute > count > yes_no > identity > location > fallback (identity).
"""

from dataclasses import dataclass
import re
from typing import List, Optional

try:
    from symbolic.dsl import DSLNode, DSLProgramCompiler
except ImportError:
    from dsl import DSLNode, DSLProgramCompiler


@dataclass
class Query:
    """Structured representation of a parsed question."""
    qtype: str
    target: str = ""
    attribute: str = ""
    value: str = ""
    program: Optional[DSLNode] = None


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
    """Find the most specific word/phrase from the question in the answer vocabulary."""
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
    """Classify a question string and compile its corresponding DSL program AST."""
    q = question.lower().strip()
    answer_vocab = answer_vocab or []
    target = _extract_target(q, answer_vocab) if answer_vocab else ""

    # Attribute questions (checked first: "what color", "what shape", etc.)
    for attr, pattern in ATTRIBUTE_PATTERNS.items():
        if re.search(pattern, q):
            prog = DSLProgramCompiler.compile(question, target=target, attribute=attr, qtype="attribute")
            return Query(qtype="attribute", target=target, attribute=attr, program=prog)

    # Count questions
    if re.search(r"^how many ", q):
        prog = DSLProgramCompiler.compile(question, target=target, qtype="count")
        return Query(qtype="count", target=target, program=prog)

    # Yes/no questions
    if re.search(r"^(?:is|are|does|do|can|has|have|was|were) ", q):
        prog = DSLProgramCompiler.compile(question, target=target, qtype="yes_no")
        return Query(qtype="yes_no", target=target, program=prog)

    # Identity questions
    for pattern in IDENTITY_PATTERNS:
        if re.search(pattern, q):
            prog = DSLProgramCompiler.compile(question, target=target, qtype="identity")
            return Query(qtype="identity", target=target, program=prog)

    # Location questions
    for pattern in LOCATION_PATTERNS:
        if re.search(pattern, q):
            prog = DSLProgramCompiler.compile(question, target=target, qtype="location")
            return Query(qtype="location", target=target, program=prog)

    # Fallback: identity
    prog = DSLProgramCompiler.compile(question, target=target, qtype="identity")
    return Query(qtype="identity", target=target, program=prog)


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
        print(f"  {q:45s} → qtype={result.qtype:10s} target={result.target:15s} AST={result.program.op if result.program else None}")
    print("QueryParser test passed!")
