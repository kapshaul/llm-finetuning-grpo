"""Rejection sampling: turn sampled reasoning candidates into SFT rows.

A candidate response is kept only if it is

1. correct (rule-based ``correctness_reward``, or ``verifier(example, response) >= 1``),
   judged independently of any format bonus,
2. structured (``format_reward == 1``),
3. readable: its *reasoning* contains no Markdown code fences (```` ``` ```` or
   ``~~~``), no paragraph longer than
   ``max_paragraph_chars``, and scores at least ``min_language_consistency`` with the
   language scorer (default: the illustrative English-word heuristic).

Code fences and paragraph length are checked in the reasoning only; the final answer may
legitimately contain code. Identical responses to the same prompt are kept once. The
thresholds are defaults of this implementation, not values disclosed by DeepSeek-R1.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from .data import validate_candidates
from .rewards import call_scorer, correctness_reward, english_word_fraction, format_reward, reasoning_text

__all__ = ["filter_candidates", "is_readable_reasoning"]

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_CODE_FENCES = ("```", "~~~")  # backtick and tilde Markdown fences
_DROPPED_KEYS = ("responses", "answer", "prompt", "response", "domain", "sample_index")


def is_readable_reasoning(reasoning: str, max_paragraph_chars: int = 2000) -> bool:
    """Structural readability check: no code fences and no overlong paragraphs."""
    if any(fence in reasoning for fence in _CODE_FENCES):
        return False
    return all(len(paragraph.strip()) <= max_paragraph_chars for paragraph in _PARAGRAPH_SPLIT_RE.split(reasoning))


def filter_candidates(
    records: list[dict],
    min_language_consistency: float = 0.9,
    max_paragraph_chars: int = 2000,
    *,
    verifier: Callable[[dict, str], float] | None = None,
    language_scorer: Callable[[str], float] | None = None,
) -> list[dict]:
    """Return accepted SFT rows ``{..metadata, prompt, response, domain, sample_index}``.

    ``answer`` and ``responses`` are removed; other candidate fields (e.g. ``id``) are
    kept as metadata. ``sample_index`` is the position of the response in ``responses``.
    The result may be empty; SFT training rejects empty datasets.
    """
    if isinstance(min_language_consistency, bool) or not isinstance(min_language_consistency, (int, float)):
        raise TypeError("min_language_consistency must be a number")
    if not math.isfinite(min_language_consistency) or not 0 <= min_language_consistency <= 1:
        raise ValueError(f"min_language_consistency must be in [0, 1], got {min_language_consistency!r}")
    if isinstance(max_paragraph_chars, bool) or not isinstance(max_paragraph_chars, int):
        raise TypeError("max_paragraph_chars must be an integer")
    if max_paragraph_chars <= 0:
        raise ValueError(f"max_paragraph_chars must be positive, got {max_paragraph_chars}")
    validate_candidates(records, require_answer=verifier is None)

    accepted: list[dict] = []
    seen: dict[str, set[str]] = {}
    for record in records:
        example = {key: value for key, value in record.items() if key != "responses"}
        metadata = {key: value for key, value in record.items() if key not in _DROPPED_KEYS}
        prompt = record["prompt"]
        prompt_seen = seen.setdefault(prompt, set())
        for index, response in enumerate(record["responses"]):
            text = response.strip()
            if text in prompt_seen:
                continue
            prompt_seen.add(text)
            if verifier is not None:
                correct = call_scorer(verifier, "verifier", example, text) >= 1.0
            else:
                correct = correctness_reward(text, record["answer"]) == 1.0
            if not correct or format_reward(text) != 1.0:
                continue
            reasoning = reasoning_text(text)
            if not is_readable_reasoning(reasoning, max_paragraph_chars):
                continue
            if language_scorer is not None:
                language = call_scorer(language_scorer, "language_scorer", reasoning)
            else:
                language = english_word_fraction(reasoning)
            if language < min_language_consistency:
                continue
            accepted.append(
                {**metadata, "prompt": prompt, "response": text, "domain": "reasoning", "sample_index": index}
            )
    return accepted
