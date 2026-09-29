import json

import pytest

from r1_grpo.data import merge_sft, read_jsonl, validate_candidates, validate_examples, write_jsonl
from r1_grpo.evaluation import evaluate_records
from r1_grpo.prompts import format_prompt
from r1_grpo.rejection import filter_candidates
from r1_grpo.rewards import (
    answer_key,
    correctness_reward,
    extract_answer,
    final_text,
    format_reward,
    language_consistency,
    normalize_answer,
    reasoning_text,
    score_response,
)

GOOD = "<think>Two plus two makes four.</think> <answer>\\boxed{4}</answer>"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_format_prompt_wraps_question_with_tag_contract():
    prompt = format_prompt("  What is 2+2?  ")
    assert "What is 2+2?" in prompt
    assert "<think>" in prompt and "</answer>" in prompt
    assert prompt.endswith("Assistant:")
    assert "{" not in prompt.replace("\\boxed{}", "")
    assert prompt.isascii() and len(prompt.encode("ascii")) < 300
    assert format_prompt("What is 2+2?", domain="reasoning") == format_prompt("What is 2+2?")


def test_format_prompt_general_domain_has_no_tag_contract():
    prompt = format_prompt("  Say hi  ", domain="general")
    assert "Say hi" in prompt and prompt.endswith("Assistant:")
    assert "User:" in prompt
    for tag in ("<think>", "</think>", "<answer>", "</answer>", "\\boxed"):
        assert tag not in prompt
    assert len(prompt.encode("ascii")) < 300


@pytest.mark.parametrize("question", ["", "   ", None])
def test_format_prompt_rejects_empty_questions(question):
    with pytest.raises((TypeError, ValueError)):
        format_prompt(question)


@pytest.mark.parametrize("domain", ["code", "", None, "Reasoning"])
def test_format_prompt_rejects_unknown_domains(domain):
    with pytest.raises(ValueError, match="domain"):
        format_prompt("What is 2+2?", domain=domain)


# ---------------------------------------------------------------------------
# Extraction and correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, expected",
    [
        (GOOD, "4"),
        ("<think>x</think><answer>The result is \\boxed{\\frac{1}{2}}.</answer>", "\\frac{1}{2}"),
        ("<think>x</think><answer>\\boxed{1} then \\boxed{2}</answer>", "2"),
        ("<think>x</think><answer> 17 </answer>", "17"),
        ("<think>try \\boxed{3}</think><answer>5</answer>", "5"),
        ("<think>try \\boxed{3}</think>", ""),
        ("<think>still going \\boxed{3} <answer>3</answer>", ""),
        ("<think>x</think><answer>\\boxed{12</answer>", ""),
        ("<think>x</think><answer>7", ""),
        ("The answer is \\boxed{ -2 }.", "-2"),
        ("42", "42"),
        ("Some text <answer>9</answer>", "9"),
        # Malformed tags anywhere in the final text invalidate the answer, even after a
        # valid block.
        ("<answer>4</answer><answer>5", ""),
        ("<think>x</think><answer>4</answer><answer>5", ""),
        ("<think>x</think><answer>4</answer></answer>", ""),
        ("<think>x</think><answer><answer>4</answer></answer>", ""),
        ("<think>x</think></answer>4", ""),
        # Reasoning reopened after the last </think> is unfinished; nothing is final.
        ("<think>x</think><answer>4</answer><think>more", ""),
    ],
)
def test_extract_answer(response, expected):
    assert extract_answer(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "<answer>4</answer><answer>5",
        "<think>x</think><answer>4</answer><answer>4",
        "<think>x</think><answer>4</answer></answer>",
    ],
)
def test_malformed_trailing_answer_tags_score_zero(response):
    assert correctness_reward(response, "4") == 0.0
    assert correctness_reward(response, "5") == 0.0


def test_extract_answer_handles_escaped_braces():
    assert extract_answer("<answer>\\boxed{\\{1, 2\\}}</answer>") == "\\{1, 2\\}"


def test_reasoning_and_final_text_split():
    response = "<think> private work </think>\n<answer> Hello! </answer>"
    assert reasoning_text(response) == "private work"
    assert final_text(response) == "Hello!"
    assert final_text("<think>unfinished") == ""
    assert reasoning_text("plain reply") == ""
    assert final_text("plain reply") == "plain reply"


@pytest.mark.parametrize(
    "response",
    [
        "<think>done</think> Hello <think>secret unfinished plan",
        "<think>done</think><answer>Hi</answer><think>secret unfinished plan",
        "<think>secret unfinished plan <answer>Hi</answer>",
        "<think>done</think><answer>Hi</answer><answer>secret",
        "<think>done</think><answer>secret",
    ],
)
def test_final_text_never_exposes_unfinished_reasoning_or_malformed_tags(response):
    assert "secret" not in final_text(response)
    assert final_text(response) == ""


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$\\dfrac{1}{2}$", "\\frac{1}{2}"),
        ("\\boxed{\\text{ New York }}", "new york"),
        ("1,000", "1000"),
        ("1{,}000", "1000"),
        ("1\\,000", "1000"),
        ("45^\\circ", "45"),
        ("50\\%", "50%"),
        ("3.", "3"),
        ("\\left( 1, 2 \\right)", "(1,2)"),
        ("x = - 5", "x=-5"),
        ("1  2", "1 2"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    "predicted, reference",
    [
        ("0.5", "\\frac{1}{2}"),
        ("\\frac12", "1/2"),
        ("2/4", "0.5"),
        ("-\\dfrac{3}{4}", "-0.75"),
        ("x = 5", "5"),
        ("1,000", "1000"),
        ("$3$", "3"),
        ("4.0", "4"),
        ("(1+2)*3", "9"),
        ("1e3", "1000"),
        ("(1, 2)", "(1,2)"),
        ("50%", ".5"),
        ("50\\%", "1/2"),
        ("12.5%", "\\frac{1}{8}"),
        ("1 + 2", "3"),
        ("2 \\times 3", "6"),
        ("\\frac {1} {2}", "0.5"),
        ("new  York", "New York"),
    ],
)
def test_equivalent_answers_are_correct(predicted, reference):
    assert correctness_reward(f"<think>w</think><answer>{predicted}</answer>", reference) == 1.0


@pytest.mark.parametrize(
    "predicted, reference",
    [
        ("50%", "50"),
        ("50", "50\\%"),
        ("1 2", "12"),
        ("12", "1 2"),
        ("1 000", "1000"),
        ("NewYork", "New York"),
    ],
)
def test_distinct_answers_are_not_merged(predicted, reference):
    assert correctness_reward(f"<think>w</think><answer>{predicted}</answer>", reference) == 0.0
    assert answer_key(predicted) != answer_key(reference)


def test_numeric_formats_share_one_answer_key():
    formats = [
        "0.5",
        ".5",
        "0.50",
        "5e-1",
        "50%",
        "50\\%",
        "$50\\%$",
        "1/2",
        "2/4",
        "\\frac{1}{2}",
        "\\dfrac12",
        "\\boxed{0.5}",
    ]
    assert {answer_key(value) for value in formats} == {"1/2"}


@pytest.mark.parametrize(
    "response",
    [
        "<think>w</think><answer>5</answer>",
        "<think>w</think><answer></answer>",
        "<think>the answer is 4</think>",
        "<think>w</think><answer>__import__('os').system('true')</answer>",
        "<think>w</think><answer>1/0</answer>",
        "<think>w</think><answer>2**9999999</answer>",
        "",
    ],
)
def test_wrong_or_malformed_answers_score_zero(response):
    assert correctness_reward(response, "4") == 0.0


def test_division_by_zero_and_huge_inputs_are_safe():
    assert answer_key("1/0") == "1/0"
    assert correctness_reward("<answer>1e999999</answer>", "1") == 0.0
    long_expression = "(" * 150 + "1" + ")" * 150
    assert correctness_reward(f"<answer>{long_expression}</answer>", "1") == 0.0


def test_answer_key_reduces_rationals():
    assert answer_key("0.50") == answer_key("\\frac{2}{4}") == "1/2"
    assert answer_key("") == ""


# ---------------------------------------------------------------------------
# Format and language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, expected",
    [
        (GOOD, 1.0),
        ("\n <think>a</think>\n\n<answer>b</answer>\n", 1.0),
        ("<think>a</think>", 0.0),
        ("<answer>b</answer>", 0.0),
        ("Sure! <think>a</think><answer>b</answer>", 0.0),
        ("<think>a</think><answer>b</answer> extra", 0.0),
        ("<think>a</think>text<answer>b</answer>", 0.0),
        ("<think> </think><answer>b</answer>", 0.0),
        ("<think>a</think><answer> </answer>", 0.0),
        ("<think>a <think> b</think><answer>c</answer>", 0.0),
        ("<think>a</think><answer>b</answer><answer>c</answer>", 0.0),
    ],
)
def test_format_reward_is_strict(response, expected):
    assert format_reward(response) == expected


def test_language_consistency_heuristic():
    assert language_consistency("<think>We add the two numbers.</think><answer>4</answer>") == 1.0
    mixed = language_consistency("<think>We add 两个数</think><answer>4</answer>")
    assert mixed == pytest.approx(2 / 5)
    assert language_consistency("<think>2 + 2 = 4</think><answer>4</answer>") == 1.0
    assert language_consistency("<think></think><answer>4</answer>") == 0.0
    # Only the reasoning is scored; the final answer may be in any language.
    assert language_consistency("<think>Simple sum.</think><answer>四</answer>") == 1.0


# ---------------------------------------------------------------------------
# score_response routing
# ---------------------------------------------------------------------------


def test_zero_stage_sums_accuracy_and_format():
    example = {"prompt": "2+2?", "answer": "4"}
    assert score_response(example, GOOD, "zero") == {"accuracy": 1.0, "format": 1.0, "total": 2.0}
    assert score_response(example, "\\boxed{4}", "zero") == {"accuracy": 1.0, "format": 0.0, "total": 1.0}
    assert score_response(example, "<think>a</think><answer>5</answer>")["total"] == 1.0


def test_reasoning_stage_sums_accuracy_and_language():
    example = {"prompt": "2+2?", "answer": "4"}
    scores = score_response(example, GOOD, "reasoning")
    assert scores == {"accuracy": 1.0, "language": 1.0, "total": 2.0}
    seen = []
    scores = score_response(example, GOOD, "reasoning", language_scorer=lambda text: seen.append(text) or 0.25)
    assert seen == ["Two plus two makes four."]
    assert scores["total"] == 1.25


def test_all_stage_reasoning_uses_accuracy_and_format():
    scores = score_response({"prompt": "2+2?", "answer": "4", "domain": "reasoning"}, GOOD, "all")
    assert scores == {"accuracy": 1.0, "format": 1.0, "total": 2.0}


def test_general_hooks_see_final_text_vs_full_response():
    calls = {}

    def helpfulness(prompt, text):
        calls["helpfulness"] = (prompt, text)
        return 0.75

    def harmlessness(prompt, text):
        calls["harmlessness"] = (prompt, text)
        return 1

    response = "<think>The user wants a greeting. Private note.</think> <answer>Hello there!</answer>"
    scores = score_response(
        {"prompt": "Say hi", "domain": "general"},
        response,
        "all",
        helpfulness=helpfulness,
        harmlessness=harmlessness,
    )
    assert calls["helpfulness"] == ("Say hi", "Hello there!")
    assert calls["harmlessness"] == ("Say hi", response)
    assert scores == {"helpfulness": 0.75, "harmlessness": 1.0, "total": 1.75}


def test_general_examples_require_hooks_and_all_stage():
    general = {"prompt": "Say hi", "domain": "general"}
    with pytest.raises(ValueError, match="helpfulness"):
        score_response(general, GOOD, "all", helpfulness=lambda p, t: 1.0)
    for stage in ("zero", "reasoning"):
        with pytest.raises(ValueError, match="general"):
            score_response(general, GOOD, stage, helpfulness=lambda p, t: 1.0, harmlessness=lambda p, t: 1.0)


def test_verifier_replaces_rule_accuracy():
    calls = []

    def verifier(example, response):
        calls.append((example["prompt"], response))
        return 1.0

    scores = score_response({"prompt": "write code"}, GOOD, "zero", verifier=verifier)
    assert calls == [("write code", GOOD)]
    assert scores["accuracy"] == 1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "high", None])
def test_callbacks_must_return_finite_numbers(value):
    with pytest.raises((TypeError, ValueError)):
        score_response({"prompt": "p", "answer": "4"}, GOOD, "zero", verifier=lambda e, r: value)


def test_score_response_rejects_bad_inputs():
    with pytest.raises(ValueError):
        score_response({"prompt": "p", "answer": "4"}, GOOD, "final")
    with pytest.raises(ValueError):
        score_response({"prompt": "p", "answer": "4", "domain": "code"}, GOOD)
    with pytest.raises(ValueError):
        score_response({"prompt": "p"}, GOOD)
    with pytest.raises(TypeError):
        score_response({"prompt": "p", "answer": "4"}, None)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_jsonl_round_trip_creates_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "data.jsonl"
    records = [{"id": 1, "prompt": "Übung?", "answer": "4"}, {"prompt": "b", "domain": "general"}]
    write_jsonl(path, records)
    assert read_jsonl(path) == records
    assert "Übung" in path.read_text(encoding="utf-8")
    write_jsonl(path, [])
    assert read_jsonl(path) == []


def test_read_jsonl_rejects_malformed_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prompt": "a"}\n\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":3:"):
        read_jsonl(path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_jsonl(path)
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl")


def test_write_jsonl_rejects_non_json_values(tmp_path):
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "x.jsonl", [{"reward": float("nan")}])
    with pytest.raises(TypeError):
        write_jsonl(tmp_path / "x.jsonl", ["not a dict"])


def test_validate_examples_accepts_valid_rows():
    validate_examples([{"id": "a", "prompt": "p", "answer": "4"}, {"prompt": "q", "domain": "general"}])
    validate_examples([{"prompt": "p", "response": "r", "domain": "general"}], supervised=True)


def test_validate_examples_optional_answer():
    unlabeled = [{"prompt": "write code"}, {"prompt": "p", "answer": "4"}, {"prompt": "q", "domain": "general"}]
    with pytest.raises(ValueError, match="answer"):
        validate_examples(unlabeled)
    validate_examples(unlabeled, require_answer=False)
    for bad_answer in (4, "", "  ", None):
        with pytest.raises(ValueError, match="answer"):
            validate_examples([{"prompt": "p", "answer": bad_answer}], require_answer=False)
    # The flag does not relax SFT rows.
    with pytest.raises(ValueError, match="response"):
        validate_examples([{"prompt": "p"}], supervised=True, require_answer=False)


@pytest.mark.parametrize(
    "records, supervised",
    [
        ([], False),
        ({"prompt": "p", "answer": "4"}, False),
        ([{"prompt": "p"}], False),
        ([{"prompt": "p", "answer": 4}], False),
        ([{"prompt": "", "answer": "4"}], False),
        ([{"prompt": 3, "answer": "4"}], False),
        ([{"prompt": "p", "answer": "4", "domain": "code"}], False),
        ([{"prompt": "p", "answer": "4", "id": 1.5}], False),
        ([{"prompt": "p", "domain": "general", "answer": 1}], False),
        (["row"], False),
        ([{"prompt": "p"}], True),
        ([{"prompt": "p", "response": "  "}], True),
        ([{"prompt": "p", "response": ["r"]}], True),
    ],
)
def test_validate_examples_rejects_malformed(records, supervised):
    with pytest.raises((TypeError, ValueError)):
        validate_examples(records, supervised=supervised)


def test_validate_candidates():
    validate_candidates([{"prompt": "p", "answer": "4", "responses": ["a", ""]}])
    validate_candidates([{"prompt": "p", "responses": ["a"]}], require_answer=False)
    for bad in (
        [{"prompt": "p", "answer": "4", "responses": []}],
        [{"prompt": "p", "answer": "4", "responses": "a"}],
        [{"prompt": "p", "answer": "4", "responses": [1]}],
        [{"prompt": "p", "responses": ["a"]}],
        [{"prompt": "p", "answer": "4", "responses": ["a"], "domain": "general"}],
    ):
        with pytest.raises(ValueError):
            validate_candidates(bad)


def test_merge_sft_tags_domain_and_provenance():
    reasoning = [{"id": "r1", "prompt": "2+2?", "response": GOOD}]
    general = [{"prompt": "Say hi", "response": "Hello", "source": "curated"}]
    merged = merge_sft(reasoning, general)
    assert merged == [
        {"id": "r1", "prompt": "2+2?", "response": GOOD, "domain": "reasoning", "source": "reasoning"},
        {"prompt": "Say hi", "response": "Hello", "source": "curated", "domain": "general"},
    ]
    assert "domain" not in reasoning[0]
    with pytest.raises(ValueError):
        merge_sft([{"prompt": "p", "response": "r", "domain": "general"}], general)
    with pytest.raises(ValueError, match="general"):
        merge_sft(reasoning, [])
    with pytest.raises(ValueError, match="reasoning"):
        merge_sft([], general)
    with pytest.raises(ValueError, match="response"):
        merge_sft([{"prompt": "p"}], general)
    with pytest.raises(ValueError, match="source"):
        merge_sft(reasoning, [{"prompt": "Say hi", "response": "Hello", "source": 3}])


# ---------------------------------------------------------------------------
# Rejection sampling
# ---------------------------------------------------------------------------


def test_filter_candidates_keeps_only_correct_structured_readable_unique():
    wrong = "<think>Two plus two is five.</think> <answer>5</answer>"
    unformatted = "Two plus two makes four, so \\boxed{4}"
    fenced = "<think>```python\nprint(2 + 2)\n```</think> <answer>4</answer>"
    tilde_fenced = "<think>Check it:\n~~~python\nprint(2 + 2)\n~~~</think> <answer>4</answer>"
    long_paragraph = "<think>" + "word " * 500 + "</think> <answer>4</answer>"
    two_paragraphs = "<think>First add.\n\nThen check.</think> <answer>4</answer>"
    chinese = "<think>我们计算二加二等于四</think> <answer>4</answer>"
    record = {
        "id": "q1",
        "prompt": "What is 2+2?",
        "answer": "4",
        "source": "unit",
        "responses": [
            GOOD,
            wrong,
            unformatted,
            fenced,
            long_paragraph,
            "  " + GOOD + "\n",
            chinese,
            two_paragraphs,
            tilde_fenced,
        ],
    }
    duplicate_prompt = {"prompt": "What is 2+2?", "answer": "4", "responses": [GOOD]}
    accepted = filter_candidates([record, duplicate_prompt])
    assert accepted == [
        {
            "id": "q1",
            "source": "unit",
            "prompt": "What is 2+2?",
            "response": GOOD,
            "domain": "reasoning",
            "sample_index": 0,
        },
        {
            "id": "q1",
            "source": "unit",
            "prompt": "What is 2+2?",
            "response": two_paragraphs,
            "domain": "reasoning",
            "sample_index": 7,
        },
    ]
    assert "answer" not in accepted[0] and "responses" not in accepted[0]


def test_filter_candidates_thresholds_and_hooks():
    code_answer = "<think>Loop once and print.</think> <answer>```python\nprint(1)\n```</answer>"
    tilde_answer = "<think>Print one value.</think> <answer>~~~python\nprint(1)\n~~~</answer>"
    record = {"prompt": "Write code", "responses": [code_answer, tilde_answer]}
    accepted = filter_candidates([record], verifier=lambda example, response: 1.0)
    assert [row["response"] for row in accepted] == [code_answer, tilde_answer]
    record = {"prompt": "Write code", "responses": [code_answer]}
    assert filter_candidates([record], verifier=lambda example, response: 0.5) == []
    assert filter_candidates([record], verifier=lambda e, r: 1.0, language_scorer=lambda text: 0.5) == []
    mixed = {"prompt": "p", "answer": "4", "responses": ["<think>We add 两个数</think> <answer>4</answer>"]}
    assert filter_candidates([mixed]) == []
    assert len(filter_candidates([mixed], min_language_consistency=0.4)) == 1
    short = {"prompt": "p", "answer": "4", "responses": ["<think>" + "a" * 50 + "</think> <answer>4</answer>"]}
    assert filter_candidates([short], max_paragraph_chars=49) == []
    with pytest.raises(ValueError):
        filter_candidates([short], min_language_consistency=1.5)
    with pytest.raises(ValueError):
        filter_candidates([])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _answers(*values):
    return [f"<think>w</think> <answer>{value}</answer>" for value in values]


def test_pass_at_1_is_averaged_not_any_correct():
    result = evaluate_records([{"prompt": "p", "answer": "4", "responses": _answers(4, 5, 5, 6)}])
    assert result["pass_at_1"] == 0.25
    assert result["consensus_accuracy"] == 0.0
    assert not any("any" in key for key in result)
    assert result["questions"][0]["num_correct"] == 1


def test_consensus_tie_break_uses_first_occurrence_and_rational_keys():
    result = evaluate_records([{"id": 7, "prompt": "p", "answer": "4", "responses": _answers(4, 5, 5, "4.0")}])
    question = result["questions"][0]
    assert question["consensus_answer"] == "4"
    assert question["consensus_votes"] == 2
    assert question["consensus_correct"] is True
    assert question["id"] == 7
    result = evaluate_records([{"prompt": "p", "answer": "4", "responses": _answers(5, 4, 4, 5)}])
    assert result["questions"][0]["consensus_answer"] == "5"
    assert result["consensus_accuracy"] == 0.0


def test_invalid_answers_can_win_consensus_and_count_as_failure():
    result = evaluate_records([{"prompt": "p", "answer": "7", "responses": ["", "<think>unfinished", *_answers(7)]}])
    question = result["questions"][0]
    assert question["consensus_answer"] is None
    assert question["consensus_correct"] is False
    assert question["pass_at_1"] == pytest.approx(1 / 3)
    assert question["invalid_answer_rate"] == pytest.approx(2 / 3)


def test_variable_group_sizes_weight_questions_equally():
    records = [
        {"prompt": "a", "answer": "1", "responses": _answers(1)},
        {"prompt": "b", "answer": "2", "responses": _answers(3, 3, 3)},
    ]
    result = evaluate_records(records)
    assert result["pass_at_1"] == 0.5  # sample-weighted would be 0.25
    assert result["consensus_accuracy"] == 0.5
    assert result["num_samples"] == 4
    assert (result["min_samples_per_question"], result["max_samples_per_question"]) == (1, 3)
    json.dumps(result)


def test_evaluate_records_with_verifier_and_validation():
    records = [{"prompt": "code", "responses": ["ok", "bad"]}]
    result = evaluate_records(records, verifier=lambda example, response: float(response == "ok"))
    assert result["pass_at_1"] == 0.5
    with pytest.raises(ValueError):
        evaluate_records(records)
    with pytest.raises(ValueError):
        evaluate_records([])
