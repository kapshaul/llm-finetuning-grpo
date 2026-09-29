"""Rule-based rewards, answer extraction and reward routing for the R1 pipeline.

Response structure: ``<think> reasoning </think> <answer> final answer </answer>``.

* Accuracy is rule-based: the final answer is extracted *only* from the text after the
  reasoning block (the ``<answer>`` block, the last balanced ``\\boxed{...}`` in it, or the
  plain final text) and compared to the reference by normalized string equality or exact
  rational arithmetic. Nothing is ever passed to ``eval``/``exec``; a small recursive
  descent parser over ``Fraction`` handles numbers, ``+ - * /``, parentheses, ``\\frac``
  and a postfix percent sign (``50%`` is ``1/2``, not ``50``). Malformed answer tags
  (unclosed, stray or nested) yield no answer, even after an earlier valid block.
* Format is strict and binary (see :func:`format_reward`).
* Language consistency is an *illustrative, replaceable* heuristic: the fraction of words in
  the reasoning written in ASCII Latin script, used as a proxy for "English". It is not a
  linguistic classifier; DeepSeek-R1 does not disclose its classifier. Supply a
  ``language_scorer`` callback for anything serious.
* General-domain preference rewards (helpfulness/harmlessness) have no built-in model; the
  paper's reward models are not available, so callers must provide callbacks.

Stage routing in :func:`score_response` (reward weights are an explicit default of plain
summation; the paper does not publish weights):

* ``zero`` (R1-Zero): accuracy + format, reasoning-domain examples only.
* ``reasoning`` (R1 reasoning-oriented RL): accuracy + language consistency.
* ``all`` (R1 all-scenario RL): reasoning examples get accuracy + format; general examples
  get ``helpfulness(prompt, final_text) + harmlessness(prompt, full_response)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
STAGES = ("zero", "reasoning", "all")
DOMAINS = ("reasoning", "general")

__all__ = [
    "ANSWER_CLOSE",
    "ANSWER_OPEN",
    "DOMAINS",
    "STAGES",
    "THINK_CLOSE",
    "THINK_OPEN",
    "answer_key",
    "call_scorer",
    "correctness_reward",
    "english_word_fraction",
    "extract_answer",
    "final_text",
    "format_reward",
    "language_consistency",
    "normalize_answer",
    "reasoning_text",
    "score_response",
]

_ANSWER_TAG_RE = re.compile(r"</?answer>")
_FORMAT_RE = re.compile(r"\A\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z", re.DOTALL)
_TEXT_WRAPPER_RE = re.compile(r"\\(?:textbf|textit|text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?")
_THOUSANDS_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_LATEX_THOUSANDS_RE = re.compile(r"(?<=\d)\\,(?=\d{3}(?!\d))")
_SYMBOL_SPACE_RE = re.compile(r"(?<!\w) | (?!\w)")
_ASSIGNMENT_RE = re.compile(r"\A[a-z]=(?=.)")
_WORD_RE = re.compile(r"[^\W\d_]+")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_MAX_RATIONAL_CHARS = 200


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


def _split_response(response: str) -> tuple[str, str]:
    """Split a response into (reasoning, final region).

    An unterminated ``<think>`` block means the whole remainder is reasoning and there is no
    final region, so no answer can leak out of unfinished reasoning. This includes a
    ``<think>`` reopened after the last ``</think>``.
    """
    open_at = response.find(THINK_OPEN)
    close_at = response.rfind(THINK_CLOSE)
    if close_at != -1:
        start = open_at + len(THINK_OPEN) if -1 < open_at < close_at else 0
        region = response[close_at + len(THINK_CLOSE) :]
        return response[start:close_at], "" if THINK_OPEN in region else region
    if open_at != -1:
        return response[open_at + len(THINK_OPEN) :], ""
    answer_at = response.find(ANSWER_OPEN)
    if answer_at != -1:
        return response[:answer_at], response[answer_at:]
    return "", response


def _answer_blocks(region: str) -> list[str] | None:
    """Return the contents of the ``<answer>`` blocks in ``region``.

    ``None`` means the tags are malformed: an unclosed, stray or nested tag anywhere makes
    the whole region invalid, so a broken trailing block cannot hide behind a valid one.
    """
    blocks: list[str] = []
    start: int | None = None
    for match in _ANSWER_TAG_RE.finditer(region):
        if match.group() == ANSWER_OPEN:
            if start is not None:
                return None
            start = match.end()
        else:
            if start is None:
                return None
            blocks.append(region[start : match.start()])
            start = None
    return None if start is not None else blocks


def reasoning_text(response: str) -> str:
    """Return the reasoning part of a response (inside ``<think>`` when present)."""
    reasoning, _ = _split_response(_require_str(response, "response"))
    return reasoning.strip()


def final_text(response: str) -> str:
    """Return the user-facing final part of a response.

    That is the last ``<answer>`` block if present, else the text after the reasoning.
    Returns ``""`` for unfinished reasoning or malformed answer tags, so unfinished
    reasoning is never shown to a helpfulness scorer.
    """
    _, region = _split_response(_require_str(response, "response"))
    blocks = _answer_blocks(region)
    if blocks is None:
        return ""
    if blocks:
        return blocks[-1].strip()
    return region.strip()


def _parse_braced(text: str, start: int) -> tuple[str, int] | None:
    """Parse a balanced ``{...}`` group starting at ``text[start] == '{'``."""
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return None


def _last_boxed(text: str) -> str:
    """Return the content of the last top-level ``\\boxed{...}``; ``""`` if it is unbalanced."""
    result = ""
    pos = 0
    while True:
        index = text.find("\\boxed", pos)
        if index == -1:
            return result
        j = index + len("\\boxed")
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "{":
            pos = j
            continue
        parsed = _parse_braced(text, j)
        if parsed is None:
            return ""
        result, pos = parsed


def extract_answer(response: str) -> str:
    """Extract the final answer from the non-reasoning part of a response.

    Priority: last complete ``<answer>`` block (its last balanced ``\\boxed`` if any), else
    the last balanced ``\\boxed`` in the final text, else the plain final text. Returns
    ``""`` for malformed answer tags (unclosed, stray or nested anywhere in the final text),
    unbalanced final boxes, or unterminated reasoning.
    """
    _, region = _split_response(_require_str(response, "response"))
    blocks = _answer_blocks(region)
    if blocks is None:
        return ""
    candidate = blocks[-1] if blocks else region
    if "\\boxed" in candidate:
        return _last_boxed(candidate).strip()
    return candidate.strip()


def format_reward(response: str) -> float:
    """Return 1.0 iff the response is exactly ``<think>R</think> <answer>A</answer>``.

    Requirements: nothing but whitespace before ``<think>`` or after ``</answer>``, each of
    the four tags appears exactly once, and both ``R`` and ``A`` are non-blank.
    """
    _require_str(response, "response")
    match = _FORMAT_RE.match(response)
    if match is None:
        return 0.0
    if any(response.count(tag) != 1 for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)):
        return 0.0
    if not match.group("think").strip() or not match.group("answer").strip():
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Answer normalization and safe rational comparison
# ---------------------------------------------------------------------------


def _unwrap_boxed(text: str) -> str:
    while text.startswith("\\boxed"):
        j = len("\\boxed")
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "{":
            break
        parsed = _parse_braced(text, j)
        if parsed is None or parsed[1] != len(text):
            break
        text = parsed[0].strip()
    return text


def normalize_answer(answer: str) -> str:
    """Canonicalize an answer string for comparison (no evaluation of any kind).

    Removes surrounding math delimiters and ``\\boxed``, ``\\text``-style wrappers,
    ``\\left``/``\\right``, degree marks, escaped dollar signs, trailing periods and
    thousands separators; lowercases; maps ``\\dfrac``/``\\tfrac`` to ``\\frac`` and
    ``\\%`` to ``%`` (percent is kept: ``50%`` is not ``50``). Whitespace (including LaTeX
    spacing) collapses to one space, which is removed next to symbols (``1 + 2`` becomes
    ``1+2``) but kept between two word characters, so ``1 2`` never turns into ``12`` and
    ``New York`` never into ``NewYork``.
    """
    text = _require_str(answer, "answer").strip()
    text = _unwrap_boxed(text)
    changed = True
    while changed:
        changed = False
        for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
            if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left) : len(text) - len(right)].strip()
                text = _unwrap_boxed(text)
                changed = True
    previous = None
    while previous != text:
        previous = text
        text = _TEXT_WRAPPER_RE.sub(r"\1", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\left(?![A-Za-z])", "", text)
    text = re.sub(r"\\right(?![A-Za-z])", "", text)
    # LaTeX digit grouping: 1{,}000 and 1\,000 are thousands separators, not spacing.
    text = text.replace("{,}", ",")
    text = _LATEX_THOUSANDS_RE.sub(",", text)
    for spacing in ("\\!", "\\,", "\\;", "\\:", "\\ ", "~"):
        text = text.replace(spacing, " ")
    text = _DEGREE_RE.sub("", text).replace("°", "")
    text = text.replace("\\%", "%").replace("\\$", "")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    text = _SYMBOL_SPACE_RE.sub("", text)
    text = text.rstrip(".")
    if _THOUSANDS_RE.fullmatch(text):
        text = text.replace(",", "")
    return text


class _ParseError(Exception):
    pass


class _RationalParser:
    """Recursive descent parser for exact rational expressions over normalized text.

    Spaces separate tokens but are never part of a number, so ``1 2`` is a parse error
    rather than ``12``. A postfix ``%`` divides by 100.
    """

    _NUMBER_RE = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:e[+-]?[0-9]{1,3})?")

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse(self) -> Fraction:
        value = self._expression()
        self._skip_space()
        if self.pos != len(self.text):
            raise _ParseError(self.text)
        return value

    def _skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1

    def _take(self, token: str) -> bool:
        self._skip_space()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._take(token):
            raise _ParseError(self.text)

    def _expression(self) -> Fraction:
        value = self._term()
        while True:
            if self._take("+"):
                value += self._term()
            elif self._take("-"):
                value -= self._term()
            else:
                return value

    def _term(self) -> Fraction:
        value = self._unary()
        while True:
            if self._take("*") or self._take("\\cdot") or self._take("\\times"):
                value *= self._unary()
            elif self._take("/") or self._take("\\div"):
                divisor = self._unary()
                if divisor == 0:
                    raise _ParseError(self.text)
                value /= divisor
            else:
                return value

    def _unary(self) -> Fraction:
        if self._take("-"):
            return -self._unary()
        if self._take("+"):
            return self._unary()
        value = self._atom()
        while self._take("%"):
            value /= 100
        return value

    def _atom(self) -> Fraction:
        for opening, closing in (("(", ")"), ("{", "}")):
            if self._take(opening):
                value = self._expression()
                self._expect(closing)
                return value
        if self._take("\\frac"):
            numerator = self._frac_argument()
            denominator = self._frac_argument()
            if denominator == 0:
                raise _ParseError(self.text)
            return numerator / denominator
        self._skip_space()
        match = self._NUMBER_RE.match(self.text, self.pos)
        if match is None:
            raise _ParseError(self.text)
        self.pos = match.end()
        return Fraction(match.group())

    def _frac_argument(self) -> Fraction:
        # Braceless \frac arguments are single digits (\frac12 is 1/2).
        if self._take("{"):
            value = self._expression()
            self._expect("}")
            return value
        if self.pos < len(self.text) and self.text[self.pos] in "0123456789":
            self.pos += 1
            return Fraction(int(self.text[self.pos - 1]))
        raise _ParseError(self.text)


def _parse_rational(text: str) -> Fraction | None:
    if not text or len(text) > _MAX_RATIONAL_CHARS:
        return None
    try:
        return _RationalParser(text).parse()
    except (_ParseError, ZeroDivisionError, ValueError, OverflowError, RecursionError):
        return None


def answer_key(answer: str) -> str:
    """Return a canonical comparison key for an extracted answer.

    Normalizes the string, drops a leading single-letter assignment (``x=``) and, when the
    remainder is an exact rational expression, replaces it by the reduced fraction
    (``0.5``, ``.5``, ``50%``, ``\\frac{1}{2}`` and ``2/4`` all map to ``1/2``). ``""`` means
    no answer.
    """
    key = _ASSIGNMENT_RE.sub("", normalize_answer(answer), count=1)
    value = _parse_rational(key)
    return str(value) if value is not None else key


def correctness_reward(response: str, answer: str) -> float:
    """Return 1.0 iff the extracted final answer matches the reference answer."""
    _require_str(answer, "answer")
    predicted = answer_key(extract_answer(response))
    reference = answer_key(answer)
    return 1.0 if predicted and predicted == reference else 0.0


# ---------------------------------------------------------------------------
# Language consistency (illustrative heuristic)
# ---------------------------------------------------------------------------


def english_word_fraction(text: str) -> float:
    """Fraction of words in ``text`` written purely in ASCII letters.

    Words are maximal runs of Unicode letters; each CJK/kana/hangul character counts as its
    own non-English word because those scripts do not separate words with spaces. Text with
    no letters (e.g. pure math) scores 1.0; blank text scores 0.0.
    """
    _require_str(text, "text")
    english = other = 0
    for token in _WORD_RE.findall(text):
        if token.isascii():
            english += 1
        else:
            other += len(_CJK_RE.findall(token)) or 1
    total = english + other
    if total == 0:
        return 1.0 if text.strip() else 0.0
    return english / total


def language_consistency(response: str) -> float:
    """English-script word fraction of the reasoning part of ``response`` (heuristic)."""
    return english_word_fraction(reasoning_text(response))


# ---------------------------------------------------------------------------
# Reward routing
# ---------------------------------------------------------------------------


def call_scorer(function: Callable[..., Any], name: str, *args: Any) -> float:
    """Call a user-supplied scoring callback and require a finite real result."""
    if not callable(function):
        raise TypeError(f"{name} callback must be callable")
    value = function(*args)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} callback must return a real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} callback returned a non-finite value: {result!r}")
    return result


def _accuracy(example: dict, response: str, verifier: Callable[..., Any] | None) -> float:
    if verifier is not None:
        return call_scorer(verifier, "verifier", example, response)
    answer = example.get("answer")
    if not isinstance(answer, str):
        raise ValueError("reasoning examples need a string 'answer' (or a verifier callback)")
    return correctness_reward(response, answer)


def score_response(
    example: dict,
    response: str,
    stage: str = "zero",
    *,
    verifier: Callable[[dict, str], float] | None = None,
    language_scorer: Callable[[str], float] | None = None,
    helpfulness: Callable[[str, str], float] | None = None,
    harmlessness: Callable[[str, str], float] | None = None,
) -> dict[str, float]:
    """Score one response for a training stage; the returned dict always has ``total``.

    Callbacks not used by the stage/domain are ignored: ``zero`` uses only rule-based
    accuracy (optionally an external ``verifier``, e.g. a sandboxed code judge) and format.
    """
    if not isinstance(example, dict):
        raise TypeError(f"example must be a dict, got {type(example).__name__}")
    _require_str(response, "response")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    domain = example.get("domain", "reasoning")
    if domain not in DOMAINS:
        raise ValueError(f"example domain must be one of {DOMAINS}, got {domain!r}")

    if domain == "general":
        if stage != "all":
            raise ValueError(f"general-domain examples are only allowed in stage 'all', not {stage!r}")
        if helpfulness is None or harmlessness is None:
            raise ValueError(
                "general-domain examples require both 'helpfulness' and 'harmlessness' callbacks; "
                "no built-in preference model is provided"
            )
        prompt = example.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("general-domain examples need a string 'prompt'")
        helpful = call_scorer(helpfulness, "helpfulness", prompt, final_text(response))
        harmless = call_scorer(harmlessness, "harmlessness", prompt, response)
        return {"helpfulness": helpful, "harmlessness": harmless, "total": helpful + harmless}

    accuracy = _accuracy(example, response, verifier)
    if stage == "reasoning":
        reasoning = reasoning_text(response)
        if language_scorer is not None:
            language = call_scorer(language_scorer, "language_scorer", reasoning)
        else:
            language = english_word_fraction(reasoning)
        return {"accuracy": accuracy, "language": language, "total": accuracy + language}
    formatted = format_reward(response)
    return {"accuracy": accuracy, "format": formatted, "total": accuracy + formatted}
