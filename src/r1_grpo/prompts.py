"""Prompt templates for R1-style training and sampling.

The wording is an original, short paraphrase, not a copy of the DeepSeek-R1 template.

* ``domain="reasoning"`` keeps the structural contract of the paper: reasoning inside
  ``<think> ... </think>`` followed by the final answer inside ``<answer> ... </answer>``,
  plus a hint to box mathematical results. It constrains only the output structure and
  does not ask for reflection or any particular problem-solving strategy.
* ``domain="general"`` is a plain User/Assistant turn asking for a direct answer, with no
  tag requirement; general-domain responses are scored by preference callbacks.
"""

from __future__ import annotations

DOMAINS = ("reasoning", "general")

PROMPT_PREFIX = (
    "A user asks a question and the assistant answers. The assistant first reasons, then "
    "gives the final answer, formatted as <think> reasoning </think> <answer> answer </answer>. "
    "Put a math result in \\boxed{} inside the answer.\n"
    "User: "
)
GENERAL_PROMPT_PREFIX = "A user asks a question and the assistant answers it directly and helpfully.\nUser: "
PROMPT_SUFFIX = "\nAssistant:"

__all__ = ["DOMAINS", "GENERAL_PROMPT_PREFIX", "PROMPT_PREFIX", "PROMPT_SUFFIX", "format_prompt"]


def format_prompt(question: str, domain: str = "reasoning") -> str:
    """Wrap a raw question in the training/sampling template for ``domain``."""
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    question = question.strip()
    if not question:
        raise ValueError("question must be a non-empty string")
    prefix = PROMPT_PREFIX if domain == "reasoning" else GENERAL_PROMPT_PREFIX
    return prefix + question + PROMPT_SUFFIX
