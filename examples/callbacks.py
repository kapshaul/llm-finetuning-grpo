"""Example reward callbacks referenced as ``'examples.callbacks:<name>'``.

IMPORTANT: these are untrained, hand-written illustrations that exist only to
show the callback signatures and to make the toy configs runnable offline.
They are NOT reward models, NOT the paper's (undisclosed) helpfulness or
harmlessness reward models, NOT a language classifier and NOT a code judge.
Replace them with real scorers for any meaningful experiment.

Signatures expected by ``r1_grpo.rewards.score_response``:

* ``verifier(example: dict, response: str) -> float``
* ``language_scorer(reasoning_text: str) -> float``
* ``helpfulness(prompt: str, final_text: str) -> float``   (sees the final answer only)
* ``harmlessness(prompt: str, full_response: str) -> float`` (sees reasoning + answer)

All callbacks must return finite floats.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z]+")

# Deliberately tiny word list: an illustration of the hook, not a safety filter.
_TOY_UNSAFE_WORDS = frozenset({"weapon", "poison", "explosive", "malware"})


def toy_helpfulness(prompt: str, final_text: str) -> float:
    """Illustrative helpfulness score in [0, 1] computed from the final answer only.

    Rewards a non-empty answer that shares at least one content word with the
    prompt. Trivially gameable; for plumbing tests only.
    """
    words = _WORD.findall(final_text.lower())
    if not words:
        return 0.0
    prompt_words = {w for w in _WORD.findall(prompt.lower()) if len(w) > 3}
    overlap = 1.0 if prompt_words & set(words) else 0.0
    return 0.5 + 0.5 * overlap


def toy_harmlessness(prompt: str, full_response: str) -> float:
    """Illustrative harmlessness score in {0, 1} over the full response.

    Returns 0.0 if any word from a tiny hard-coded list appears anywhere in the
    response (reasoning included), else 1.0. Not a safety classifier.
    """
    del prompt  # the toy rule inspects the response only
    words = set(_WORD.findall(full_response.lower()))
    return 0.0 if words & _TOY_UNSAFE_WORDS else 1.0


def ascii_letter_ratio(reasoning_text: str) -> float:
    """Illustrative language scorer: fraction of alphabetic characters that are ASCII.

    A crude proxy for "reasoning is written in English". It cannot tell English
    from other Latin-script languages. Returns 0.0 for text without letters.
    """
    letters = [c for c in reasoning_text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def multiple_choice_verifier(example: dict, response: str) -> float:
    """Illustrative rule-based verifier for single-letter multiple-choice answers.

    Compares the extracted final answer (e.g. ``\\boxed{B}``) with
    ``example['answer']`` case-insensitively, ignoring surrounding punctuation.
    Generated text is only parsed, never executed.
    """
    from r1_grpo.rewards import extract_answer

    predicted = extract_answer(response).strip().strip("().").upper()
    expected = str(example.get("answer", "")).strip().strip("().").upper()
    return 1.0 if predicted and predicted == expected else 0.0
