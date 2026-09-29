"""Offline tests for the command-line interface, configs, toy data and example callbacks."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from r1_grpo import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "examples" / "data"

RL_STAGES = {"zero", "reasoning", "all"}
SFT_STAGES = {"cold_start", "mixed_sft", "distill"}
ALLOWED_CONFIG_KEYS = {
    "backend",
    "stage",
    "model_name_or_path",
    "dataset",
    "output_dir",
    "seed",
    "device",
    "learning_rate",
    "batch_size",
    "max_steps",
    "epochs",
    "group_size",
    "update_epochs",
    "clip_epsilon",
    "beta",
    "reduction",
    "max_prompt_tokens",
    "max_new_tokens",
    "max_seq_length",
    "temperature",
    "top_p",
    "num_samples",
    "max_grad_norm",
    "callbacks",
}
CALLBACK_NAMES = {"verifier", "language_scorer", "helpfulness", "harmlessness"}
ALL_CONFIGS = sorted(CONFIG_DIR.rglob("*.json"))


def test_evaluate_uses_external_verifier_without_answer(tmp_path, monkeypatch, capsys):
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, [{"prompt": "Choose a letter", "expected": "B", "responses": ["B", "B", "A"]}])
    specs = []

    def load(spec):
        specs.append(spec)
        return lambda row, response: float(response == row["expected"])

    monkeypatch.setattr(cli, "load_callable", load)
    assert cli.main(["evaluate", "--input", str(path), "--verifier", "judge:check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert specs == ["judge:check"]
    assert result["pass_at_1"] == pytest.approx(2 / 3)
    assert result["consensus_accuracy"] == 1.0


def test_numerical_failure_is_reported_without_traceback(tmp_path, monkeypatch, capsys):
    from r1_grpo import training

    path = tmp_path / "config.json"
    path.write_text("{}")

    def fail(_):
        raise FloatingPointError("GRPO likelihood ratio overflowed")

    monkeypatch.setattr(training, "train_grpo", fail)
    assert cli.main(["grpo", "--config", str(path)]) == 1
    error = capsys.readouterr().err
    assert "likelihood ratio overflowed" in error
    assert "Traceback" not in error


def _config(relative: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / relative).read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl_raw(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _roundtrip(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _assert_finite_numbers(obj: Any) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        assert math.isfinite(obj), obj
    elif isinstance(obj, dict):
        for value in obj.values():
            _assert_finite_numbers(value)
    elif isinstance(obj, list):
        for value in obj:
            _assert_finite_numbers(value)


def _answer(value: str) -> str:
    return f"<think>\nWork it out step by step.\n</think>\n<answer>\n\\boxed{{{value}}}\n</answer>"


# ---------------------------------------------------------------------------
# Parser, entry points, help
# ---------------------------------------------------------------------------


def test_help_lists_commands_without_importing_torch() -> None:
    code = (
        "import sys\n"
        "from r1_grpo import cli\n"
        "for argv in (['--help'], ['grpo', '--help'], ['reject', '--help']):\n"
        "    try:\n"
        "        cli.main(argv)\n"
        "    except SystemExit as exc:\n"
        "        assert exc.code == 0, exc.code\n"
        "print('TORCH_LOADED=' + str('torch' in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "TORCH_LOADED=False" in result.stdout
    for command in ("sft", "grpo", "sample", "reject", "mix", "evaluate", "smoke"):
        assert command in result.stdout
    assert "--min-language-consistency" in result.stdout


def test_python_dash_m_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "r1_grpo", "--version"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "r1-grpo" in result.stdout


def test_subcommand_is_required() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["reject", "--input", "a", "--output", "b", "--min-language-consistency", "1.5"],
        ["reject", "--input", "a", "--output", "b", "--min-language-consistency", "high"],
        ["reject", "--input", "a", "--output", "b", "--max-paragraph-chars", "0"],
        ["reject", "--input", "a", "--output", "b", "--max-paragraph-chars", "-3"],
        ["sample", "--config", "c.json"],
        ["grpo"],
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Friendly errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["sft", "grpo"])
def test_missing_config_reports_friendly_error(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main([command, "--config", "does_not_exist.json"]) == 1
    err = capsys.readouterr().err
    assert "config file not found" in err
    assert "current directory" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("content", "message"),
    [("{not json", "not valid JSON"), ("[1, 2]", "must contain a JSON object")],
)
def test_malformed_config_reports_friendly_error(
    content: str, message: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = tmp_path / "bad.json"
    config.write_text(content, encoding="utf-8")
    output = tmp_path / "out.jsonl"
    assert cli.main(["sample", "--config", str(config), "--output", str(output)]) == 1
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err
    assert not output.exists()


@pytest.mark.parametrize(
    ("command", "stage"),
    [("grpo", "bogus"), ("grpo", "cold_start"), ("sft", "zero"), ("sft", "bogus")],
)
def test_wrong_stage_for_command_fails_cleanly(
    command: str, stage: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dataset = DATA_DIR / ("math_rl.jsonl" if command == "grpo" else "cold_start_sft.jsonl")
    config = tmp_path / "config.json"
    output_dir = tmp_path / "run"
    payload = {
        "backend": "tiny",
        "stage": stage,
        "dataset": str(dataset),
        "output_dir": str(output_dir),
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    assert cli.main([command, "--config", str(config)]) == 1
    err = capsys.readouterr().err
    assert f"r1-grpo {command}: error:" in err
    assert "Traceback" not in err


def test_traceback_flag_shows_traceback(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    config = tmp_path / "bad.json"
    config.write_text("[]", encoding="utf-8")
    assert cli.main(["grpo", "--config", str(config), "--traceback"]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_reject_missing_input_is_friendly(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    missing = tmp_path / "missing.jsonl"
    rc = cli.main(["reject", "--input", str(missing), "--output", str(tmp_path / "out.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "candidate file not found" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# Callback resolution
# ---------------------------------------------------------------------------


def test_load_callable_resolves_example_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    helpfulness = cli.load_callable("examples.callbacks:toy_helpfulness")
    assert callable(helpfulness)
    assert helpfulness("Name a primary color.", "Blue is a primary color.") > 0.0


@pytest.mark.parametrize(
    "spec",
    [
        "no_colon_here",
        "a:b:c",
        ":function",
        "json:",
        "json:does_not_exist",
        "json:__doc__",
        "r1_grpo_no_such_module_xyz:function",
    ],
)
def test_load_callable_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(cli.CLIError):
        cli.load_callable(spec)


# ---------------------------------------------------------------------------
# Data commands: reject, mix, evaluate
# ---------------------------------------------------------------------------


def test_reject_filters_candidates_like_the_library(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from r1_grpo.data import read_jsonl
    from r1_grpo.rejection import filter_candidates

    source = DATA_DIR / "candidates.jsonl"
    output = tmp_path / "nested" / "reasoning_sft.jsonl"
    rc = cli.main(
        [
            "reject",
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-language-consistency",
            "0.5",
            "--max-paragraph-chars",
            "500",
        ]
    )
    assert rc == 0
    rows = _read_jsonl_raw(output)
    expected = filter_candidates(read_jsonl(source), min_language_consistency=0.5, max_paragraph_chars=500)
    assert rows == _roundtrip(expected)

    # Toy candidates: 10 responses; exactly one correct, clean response survives for each
    # prompt (incorrect answers, the code fence, the unstructured reply and the duplicate
    # are rejected).
    assert len(rows) == 3, rows
    input_prompts = {record["prompt"] for record in read_jsonl(source)}
    for row in rows:
        assert row["domain"] == "reasoning"
        assert row["prompt"] in input_prompts
        assert "responses" not in row
        assert "```" not in row["response"]
        for wrong in ("\\boxed{13}", "\\boxed{6}", "\\boxed{10}"):
            assert wrong not in row["response"]
    assert len({(row["prompt"], row["response"]) for row in rows}) == len(rows)

    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 3
    assert summary["prompts"] == 3
    assert summary["candidate_responses"] == 10


def test_reject_routes_custom_verifier_and_warns_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module_dir = tmp_path / "callbacks"
    module_dir.mkdir()
    (module_dir / "r1_grpo_test_reject_all.py").write_text(
        "CALLS = []\n\ndef judge(example, response):\n    CALLS.append(example['prompt'])\n    return 0.0\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))
    output = tmp_path / "empty.jsonl"
    rc = cli.main(
        [
            "reject",
            "--input",
            str(DATA_DIR / "candidates.jsonl"),
            "--output",
            str(output),
            "--verifier",
            "r1_grpo_test_reject_all:judge",
        ]
    )
    assert rc == 0
    assert output.exists()
    assert _read_jsonl_raw(output) == []
    captured = capsys.readouterr()
    assert "no responses were accepted" in captured.err
    assert json.loads(captured.out)["accepted"] == 0

    import r1_grpo_test_reject_all

    assert r1_grpo_test_reject_all.CALLS, "the verifier callback was never called"


def test_reject_bad_callback_spec_is_friendly(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = cli.main(
        [
            "reject",
            "--input",
            str(DATA_DIR / "candidates.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--language-scorer",
            "not-a-spec",
        ]
    )
    assert rc == 1
    assert "module:function" in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_mix_merges_reasoning_and_general(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from r1_grpo.data import merge_sft, read_jsonl, validate_examples

    reasoning = DATA_DIR / "cold_start_sft.jsonl"
    general = DATA_DIR / "general_sft.jsonl"
    output = tmp_path / "mixed" / "mixed_sft.jsonl"
    rc = cli.main(["mix", "--reasoning", str(reasoning), "--general", str(general), "--output", str(output)])
    assert rc == 0
    rows = _read_jsonl_raw(output)
    assert rows == _roundtrip(merge_sft(read_jsonl(reasoning), read_jsonl(general)))
    assert len(rows) == len(read_jsonl(reasoning)) + len(read_jsonl(general))
    validate_examples(rows, supervised=True)
    assert json.loads(capsys.readouterr().out)["total"] == len(rows)


def test_mix_rejects_empty_rejection_output_clearly(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    empty = tmp_path / "reasoning_sft.jsonl"
    empty.write_text("", encoding="utf-8")
    general = DATA_DIR / "general_sft.jsonl"
    output = tmp_path / "mixed.jsonl"
    rc = cli.main(["mix", "--reasoning", str(empty), "--general", str(general), "--output", str(output)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "rejection sampling accepted no responses" in err
    assert "Traceback" not in err
    assert not output.exists()


def test_evaluate_writes_the_library_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from r1_grpo.evaluation import evaluate_records

    records = [
        # 1 of 4 samples correct: contributes 0.25 to pass@1, not 1.0.
        {
            "id": "a",
            "prompt": "What is 1 + 1?",
            "answer": "2",
            "responses": [_answer("2"), _answer("3"), _answer("3"), _answer("4")],
        },
        # Different group size: 2 of 2 correct.
        {"id": "b", "prompt": "What is 2 + 2?", "answer": "4", "responses": [_answer("4")] * 2},
    ]
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, records)
    report_path = tmp_path / "reports" / "report.json"
    assert cli.main(["evaluate", "--input", str(source), "--output", str(report_path)]) == 0

    expected = _roundtrip(evaluate_records(records))
    assert json.loads(report_path.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == expected
    _assert_finite_numbers(expected)

    # Per-question averaged pass@1 is (0.25 + 1.0) / 2; the any-correct rate would be 1.0
    # and the sample-weighted rate 0.5.
    assert math.isclose(expected["pass_at_1"], 0.625)
    assert expected["num_samples"] == 6


def test_evaluate_without_output_prints_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, [{"prompt": "1+1?", "answer": "2", "responses": [_answer("2")]}])
    assert cli.main(["evaluate", "--input", str(source)]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidates.jsonl"]


# ---------------------------------------------------------------------------
# Configs and toy data
# ---------------------------------------------------------------------------


def test_config_files_exist() -> None:
    names = {path.relative_to(CONFIG_DIR).as_posix() for path in ALL_CONFIGS}
    assert {
        "r1_zero.json",
        "r1_stage1_cold_start_sft.json",
        "r1_stage2_reasoning_rl.json",
        "r1_stage3_sample.json",
        "r1_stage3_mixed_sft.json",
        "r1_stage4_all_rl.json",
        "distill_sft.json",
        "eval_toy.json",
        "hf/eval_paper_v1.json",
        "hf/r1_zero_qwen2.5_math_1.5b.json",
        "hf/distill_sft_qwen2.5_math_1.5b.json",
    } <= names


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_config_schema(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = cli.load_config(path)
    assert set(config) <= ALLOWED_CONFIG_KEYS, set(config) - ALLOWED_CONFIG_KEYS
    assert config["backend"] in {"tiny", "hf"}
    assert "dataset" in config
    if config["backend"] == "hf":
        assert "model_name_or_path" in config

    stage = config.get("stage")
    if stage in RL_STAGES:
        # On-policy sampling: the sampled distribution must match the log-probs in the ratio.
        assert config["temperature"] == 1.0
        assert config["top_p"] == 1.0
        assert config["max_prompt_tokens"] + config["max_new_tokens"] <= config["max_seq_length"]
        assert config["reduction"] == "sequence"
    elif stage in SFT_STAGES:
        assert config["epochs"] >= 1
    else:
        assert stage is None, f"unknown stage {stage!r}"
        # Sampling / evaluation configs.
        assert config["temperature"] == 0.6
        assert config["top_p"] == 0.95
        assert 4 <= config["num_samples"] <= 64
        assert config["max_prompt_tokens"] + config["max_new_tokens"] <= config["max_seq_length"]
        assert "output_dir" not in config

    callbacks = config.get("callbacks", {})
    assert set(callbacks) <= CALLBACK_NAMES
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    for spec in callbacks.values():
        assert callable(cli.load_callable(spec))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_config_datasets_exist_and_validate(path: Path) -> None:
    from r1_grpo.data import read_jsonl, validate_examples

    config = cli.load_config(path)
    dataset = config["dataset"]
    if dataset.startswith("runs/"):
        # Produced by an earlier pipeline step (see README).
        return
    records = read_jsonl(REPO_ROOT / dataset)
    validate_examples(records, supervised=config.get("stage") in SFT_STAGES)


def test_r1_pipeline_configs_chain_correctly() -> None:
    zero = _config("r1_zero.json")
    stage1 = _config("r1_stage1_cold_start_sft.json")
    stage2 = _config("r1_stage2_reasoning_rl.json")
    sample = _config("r1_stage3_sample.json")
    stage3 = _config("r1_stage3_mixed_sft.json")
    stage4 = _config("r1_stage4_all_rl.json")
    distill = _config("distill_sft.json")
    eval_toy = _config("eval_toy.json")

    # R1-Zero: RL straight from the (fresh tiny) base, rule rewards only.
    assert zero["stage"] == "zero"
    assert "model_name_or_path" not in zero
    assert not zero.get("callbacks")

    assert stage1["stage"] == "cold_start"
    assert "model_name_or_path" not in stage1
    assert stage2["stage"] == "reasoning"
    assert stage2["model_name_or_path"] == stage1["output_dir"]
    assert sample["model_name_or_path"] == stage2["output_dir"]

    # Stage 3 SFT restarts from the original base (same seed), for 2 epochs.
    assert stage3["stage"] == "mixed_sft"
    assert "model_name_or_path" not in stage3
    assert stage3["seed"] == stage1["seed"]
    assert stage3["epochs"] == 2
    assert stage3["dataset"] == "runs/r1-stage3/mixed_sft.jsonl"

    assert stage4["stage"] == "all"
    assert stage4["model_name_or_path"] == stage3["output_dir"]
    assert {"helpfulness", "harmlessness"} <= set(stage4["callbacks"])

    # Distillation: SFT only, on the curated stage-3 data.
    assert distill["stage"] == "distill"
    assert distill["dataset"] == stage3["dataset"]
    assert eval_toy["model_name_or_path"] == stage4["output_dir"]

    output_dirs = [c["output_dir"] for c in (zero, stage1, stage2, stage3, stage4, distill)]
    assert len(set(output_dirs)) == len(output_dirs)


def test_paper_v1_eval_and_hf_substitute_configs() -> None:
    paper = _config("hf/eval_paper_v1.json")
    assert paper["temperature"] == 0.6
    assert paper["top_p"] == 0.95
    assert paper["max_new_tokens"] == 32768
    assert paper["num_samples"] == 16

    for name in ("hf/r1_zero_qwen2.5_math_1.5b.json", "hf/distill_sft_qwen2.5_math_1.5b.json"):
        config = _config(name)
        assert config["backend"] == "hf"
        assert config["model_name_or_path"] == "Qwen/Qwen2.5-Math-1.5B"


def test_readme_commands_reference_existing_configs() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"--config (configs/\S+\.json)", readme))
    assert referenced, "README should show config-driven commands"
    for relative in referenced:
        assert (REPO_ROOT / relative).is_file(), relative
    for command in ("sft", "grpo", "sample", "reject", "mix", "evaluate", "smoke"):
        assert f"r1-grpo {command}" in readme
    assert "uv sync --extra dev" in readme


def test_toy_sft_and_candidate_data_are_well_formed() -> None:
    from r1_grpo.data import read_jsonl, validate_examples

    for name in ("cold_start_sft.jsonl", "general_sft.jsonl"):
        validate_examples(read_jsonl(DATA_DIR / name), supervised=True)
    for record in read_jsonl(DATA_DIR / "candidates.jsonl"):
        assert isinstance(record["prompt"], str) and isinstance(record["answer"], str)
        assert record["responses"] and all(isinstance(r, str) for r in record["responses"])


# ---------------------------------------------------------------------------
# Example callbacks (untrained illustrations)
# ---------------------------------------------------------------------------


def test_example_callbacks_behave_as_documented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from examples import callbacks

    assert callbacks.toy_helpfulness("Name a color.", "") == 0.0
    helpful = callbacks.toy_helpfulness("Name a primary color.", "Blue is a primary color.")
    assert 0.0 < helpful <= 1.0

    clean = "<think>\nplan a birthday party\n</think>\n<answer>\nBalloons.\n</answer>"
    unsafe_reasoning = "<think>\nmention a weapon here\n</think>\n<answer>\nBalloons.\n</answer>"
    assert callbacks.toy_harmlessness("Party ideas?", clean) == 1.0
    # Harmlessness sees the whole response, including the reasoning.
    assert callbacks.toy_harmlessness("Party ideas?", unsafe_reasoning) == 0.0

    assert callbacks.ascii_letter_ratio("plain English words") == 1.0
    assert callbacks.ascii_letter_ratio("12345 !!") == 0.0
    assert math.isclose(callbacks.ascii_letter_ratio("abcdé"), 0.8)

    example = {"prompt": "Pick one: (A) 1 (B) 2", "answer": "B"}
    assert callbacks.multiple_choice_verifier(example, _answer("b")) == 1.0
    assert callbacks.multiple_choice_verifier(example, _answer("A")) == 0.0


# ---------------------------------------------------------------------------
# End-to-end through the CLI (tiny offline fixture)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sft_command_trains_and_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    config = _config("r1_stage1_cold_start_sft.json")
    config["dataset"] = str(DATA_DIR / "cold_start_sft.jsonl")
    config["output_dir"] = str(tmp_path / "cold-start")
    config["epochs"] = 1
    config_path = tmp_path / "sft.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert cli.main(["sft", "--config", str(config_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert isinstance(summary, dict)
    _assert_finite_numbers(summary)
    assert any((tmp_path / "cold-start").iterdir())

    # A second run into the same output directory must not silently overwrite it.
    assert cli.main(["sft", "--config", str(config_path)]) == 1
    assert "error" in capsys.readouterr().err


@pytest.mark.slow
def test_smoke_command_runs_offline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["smoke", "--output-dir", str(tmp_path / "smoke")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert isinstance(summary, dict) and summary
    _assert_finite_numbers(summary)
    assert (tmp_path / "smoke").is_dir()
