"""Evaluation of sampled candidates: averaged pass@1 and majority-vote consensus.

For each question with ``k`` sampled responses (``k`` may differ across questions):

* ``pass@1`` is the fraction of its samples that are correct; the reported ``pass_at_1``
  is the unweighted mean over questions, so questions with more samples carry no extra
  weight. This is *not* "any sample correct" (pass@k), which is intentionally not reported.
* Consensus takes a majority vote over canonical answer keys (normalized answers, with
  exact rationals reduced so ``0.5`` and ``1/2`` agree). Ties go to the answer seen first.
  Empty or unextractable answers form their own bucket; if it wins, the question counts as
  a consensus failure.

The paper's evaluation defaults (temperature 0.6, top-p 0.95, 16 samples) belong to the
sampling step that produces these records, not to this function.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean

from .data import validate_candidates
from .rewards import answer_key, call_scorer, correctness_reward, extract_answer, format_reward

__all__ = ["evaluate_records"]


def evaluate_records(records: list[dict], *, verifier: Callable[[dict, str], float] | None = None) -> dict:
    """Evaluate candidate rows ``{prompt, answer, responses, [id]}``.

    ``verifier(example, response)`` optionally replaces rule-based correctness; a sample
    counts as correct when it returns at least 1.0.
    """
    validate_candidates(records, require_answer=verifier is None)
    questions = []
    for index, record in enumerate(records):
        example = {key: value for key, value in record.items() if key != "responses"}
        responses = record["responses"]
        answers = [extract_answer(response) for response in responses]
        if verifier is not None:
            correct = [call_scorer(verifier, "verifier", example, response) >= 1.0 for response in responses]
        else:
            correct = [correctness_reward(response, record["answer"]) == 1.0 for response in responses]

        votes: dict[str, int] = {}
        first_index: dict[str, int] = {}
        for sample_index, answer in enumerate(answers):
            key = answer_key(answer)
            votes[key] = votes.get(key, 0) + 1
            first_index.setdefault(key, sample_index)
        winner = None
        for key, count in votes.items():  # insertion order == first occurrence
            if winner is None or count > votes[winner]:
                winner = key
        winner_index = first_index[winner]
        consensus_valid = winner != ""

        detail = {
            "index": index,
            "prompt": record["prompt"],
            "num_samples": len(responses),
            "num_correct": sum(correct),
            "pass_at_1": sum(correct) / len(responses),
            "consensus_answer": answers[winner_index].strip() if consensus_valid else None,
            "consensus_votes": votes[winner],
            "consensus_correct": consensus_valid and correct[winner_index],
            "format_rate": fmean(format_reward(response) for response in responses),
            "invalid_answer_rate": sum(1 for answer in answers if not answer_key(answer)) / len(responses),
            "extracted_answers": answers,
            "sample_correct": correct,
        }
        if "id" in record:
            detail["id"] = record["id"]
        if "answer" in record:
            detail["reference_answer"] = record["answer"]
        questions.append(detail)

    sample_counts = [detail["num_samples"] for detail in questions]
    return {
        "num_questions": len(questions),
        "num_samples": sum(sample_counts),
        "min_samples_per_question": min(sample_counts),
        "max_samples_per_question": max(sample_counts),
        "mean_samples_per_question": fmean(sample_counts),
        "pass_at_1": fmean(detail["pass_at_1"] for detail in questions),
        "consensus_accuracy": fmean(1.0 if detail["consensus_correct"] else 0.0 for detail in questions),
        "format_rate": fmean(detail["format_rate"] for detail in questions),
        "invalid_answer_rate": fmean(detail["invalid_answer_rate"] for detail in questions),
        "questions": questions,
    }
