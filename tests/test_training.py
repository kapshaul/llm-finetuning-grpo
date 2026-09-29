"""Offline tests for model backends and training loops (tiny GRU fixture only)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from r1_grpo import models, training
from r1_grpo.prompts import format_prompt

SFT_ROWS = [
    {"id": "a", "prompt": "What is 1 + 1?", "response": "<think>1 + 1 = 2</think>\n<answer>2</answer>"},
    {"id": "b", "prompt": "What is 2 + 2?", "response": "<think>2 + 2 = 4</think>\n<answer>4</answer>"},
]
RL_ROWS = [
    {"id": "a", "prompt": "What is 1 + 1?", "answer": "2"},
    {"id": "b", "prompt": "What is 2 + 2?", "answer": "4"},
]
LENGTHS = {"max_prompt_tokens": 4096, "max_seq_length": 8192}
EOS = models.ByteTokenizer.eos_token_id


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def _forced_policy(token_id: int) -> models.Policy:
    """Tiny policy whose next-token distribution puts ~all mass on ``token_id``."""
    policy = models.load_policy("tiny", None)
    with torch.no_grad():
        policy.model.head.weight.zero_()
        policy.model.head.bias.fill_(-50.0)
        policy.model.head.bias[token_id] = 50.0
    return policy


# ---------------------------------------------------------------------------
# Tokenizer, log-probs, sampling
# ---------------------------------------------------------------------------
def test_byte_tokenizer_roundtrip_and_specials():
    tok = models.ByteTokenizer()
    ids = tok.encode("héllo", add_bos=True)
    assert ids[0] == tok.bos_token_id
    assert tok.decode(ids) == "héllo"
    assert tok.decode(ids + [tok.eos_token_id], skip_special_tokens=False) == "<bos>héllo<eos>"


def test_completion_log_probs_shift_masks_and_padding_immunity():
    policy = models.load_policy("tiny", None, seed=0)
    prompts = [[257, 10, 11, 12], [257, 20]]
    completions = [[30, 31, EOS], [40]]
    log_probs, mask = models.completion_log_probs(policy, prompts, completions)

    assert mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    assert log_probs[1, 1:].abs().sum().item() == 0.0
    for row, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        ids = torch.tensor([prompt + completion])
        logits, _ = policy.model(ids[:, :-1])
        logsm = torch.log_softmax(logits.float(), dim=-1)[0]
        expected = torch.stack([logsm[len(prompt) - 1 + t, token] for t, token in enumerate(completion)])
        torch.testing.assert_close(log_probs[row, : len(completion)], expected)

    # Real PyTorch gradients flow from completion log-probs to every parameter.
    (log_probs * mask).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


def test_sampling_stops_at_eos_and_keeps_it():
    policy = _forced_policy(EOS)
    samples = models.sample_completions(policy, [257, 65], 3, 10, 1.0, 1.0, torch.Generator().manual_seed(0))
    assert [s.token_ids for s in samples] == [[EOS]] * 3
    assert all(s.stopped_on_eos and s.text == "" for s in samples)


def test_truncation_appends_no_eos_and_keeps_exact_sampled_ids():
    policy = _forced_policy(255)  # a lone 0xFF byte is invalid UTF-8
    samples = models.sample_completions(policy, [257, 65], 2, 5, 1.0, 1.0, torch.Generator().manual_seed(0))
    for sample in samples:
        assert sample.token_ids == [255] * 5
        assert not sample.stopped_on_eos
        # Decoding then re-tokenizing would NOT recover the sampled actions.
        assert policy.encode_text(sample.text) != sample.token_ids


def test_top_p_restricts_support_and_full_support_at_one():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).repeat(2000, 1)
    gen = torch.Generator().manual_seed(0)
    assert set(models.sample_next_tokens(logits, 1.0, 0.5, gen).tolist()) == {0}
    assert set(models.sample_next_tokens(logits, 1.0, 0.8, gen).tolist()) == {0, 1}
    assert set(models.sample_next_tokens(logits, 1.0, 1.0, gen).tolist()) == {0, 1, 2}
    with pytest.raises(ValueError):
        models.sample_next_tokens(logits, 0.0, 1.0, gen)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def test_train_sft_updates_saves_and_reloads(tmp_path):
    out = tmp_path / "sft"
    cfg = {
        "stage": "cold_start",
        "dataset": _write(tmp_path / "sft.jsonl", SFT_ROWS),
        "output_dir": str(out),
        "learning_rate": 1e-2,
        **LENGTHS,
    }
    summary, policy = training._train_sft(cfg)

    assert summary["steps"] == 4  # 2 examples x 2 epochs, batch size 1
    assert summary["epochs"] == summary["completed_epochs"] == 2
    assert summary["parameter_delta_l2"] > 0
    assert math.isfinite(summary["final_loss"])
    for name in (models.TINY_WEIGHTS_FILE, models.TINY_CONFIG_FILE, training.METADATA_FILE):
        assert (out / name).is_file()
    metadata = json.loads((out / training.METADATA_FILE).read_text())
    assert metadata["stage"] == "cold_start"
    assert metadata["lineage"][0]["stage"] == "base"
    metrics = json.loads((out / training.METRICS_FILE).read_text())
    assert len(metrics["history"]) == 4

    reloaded = models.load_policy("tiny", str(out))
    assert models.parameter_l2_distance(reloaded.model, policy.model) == 0.0
    fresh = models.load_policy("tiny", None, seed=42)
    assert models.parameter_l2_distance(reloaded.model, fresh.model) > 0

    # The saved checkpoint becomes the next stage's model_name_or_path.
    with pytest.raises(FileExistsError):
        training.train_sft({**cfg, "model_name_or_path": str(out)})


def test_sft_masks_prompt_and_supervises_eos(tmp_path, monkeypatch):
    seen = []
    real = models.completion_log_probs

    def spy(policy, prompts, completions):
        seen.append((prompts, completions))
        return real(policy, prompts, completions)

    rows = SFT_ROWS + [{"id": "g", "prompt": "Say hello.", "response": "Hello!", "domain": "general"}]
    monkeypatch.setattr(training.models, "completion_log_probs", spy)
    training.train_sft(
        {
            "stage": "distill",
            "dataset": _write(tmp_path / "sft.jsonl", rows),
            "output_dir": str(tmp_path / "out"),
            "epochs": 1,
            **LENGTHS,
        }
    )
    tok = models.ByteTokenizer()
    # Each row uses its own domain's template (general rows get no think instruction).
    expected = {
        (
            tuple(tok.encode(format_prompt(r["prompt"], domain=r.get("domain", "reasoning")), add_bos=True)),
            tuple(tok.encode(r["response"]) + [EOS]),
        )
        for r in rows
    }
    got = {(tuple(p[0]), tuple(c[0])) for p, c in seen}
    assert got == expected


def test_sft_summary_distinguishes_requested_and_completed_epochs(tmp_path):
    summary = training.train_sft(
        {
            "stage": "cold_start",
            "dataset": _write(tmp_path / "sft.jsonl", SFT_ROWS),
            "output_dir": str(tmp_path / "out"),
            "epochs": 2,
            "max_steps": 3,
            **LENGTHS,
        }
    )
    assert summary["steps"] == 3
    assert summary["epochs"] == 2
    assert summary["steps_per_epoch"] == 2
    assert summary["completed_epochs"] == 1


def test_sft_fails_when_truncation_drops_all_labels(tmp_path):
    prompt_len = len(models.ByteTokenizer().encode(format_prompt(SFT_ROWS[0]["prompt"]), add_bos=True))
    other_len = len(models.ByteTokenizer().encode(format_prompt(SFT_ROWS[1]["prompt"]), add_bos=True))
    limit = max(prompt_len, other_len)
    with pytest.raises(ValueError, match="no completion labels"):
        training.train_sft(
            {
                "stage": "cold_start",
                "dataset": _write(tmp_path / "sft.jsonl", SFT_ROWS),
                "output_dir": str(tmp_path / "out"),
                "max_prompt_tokens": limit,
                "max_seq_length": limit,
            }
        )


def test_refuses_to_overwrite_nonempty_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("user file")
    with pytest.raises(FileExistsError):
        training.train_sft(
            {"stage": "cold_start", "dataset": _write(tmp_path / "d.jsonl", SFT_ROWS), "output_dir": str(out)}
        )
    assert (out / "keep.txt").read_text() == "user file"


@pytest.mark.parametrize(
    "override",
    [
        {"unknown_key": 1},
        {"stage": "zero"},
        {"learning_rate": -1.0},
        {"learning_rate": float("nan")},
        {"batch_size": True},
        {"epochs": 0},
        {"backend": "other"},
    ],
)
def test_sft_config_validation(tmp_path, override):
    cfg = {
        "stage": "cold_start",
        "dataset": _write(tmp_path / "d.jsonl", SFT_ROWS),
        "output_dir": str(tmp_path / "out"),
        **override,
    }
    with pytest.raises(ValueError):
        training.train_sft(cfg)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"stage": "cold_start"},
        {"temperature": 0.6},
        {"top_p": 0.95},
        {"top_p": 1.5},
        {"reduction": "mean"},
        {"beta": -0.1},
        {"group_size": 0},
        {"group_size": 1},
        {"clip_epsilon": 1.0},
        {"clip_epsilon": 1.5},
        {"max_steps": None},
        {"callbacks": {"helpfulness": "math:sqrt"}},
        {"callbacks": {"verifier": "no_colon"}},
        {"callbacks": {"reward_model": "math:sqrt"}},
    ],
)
def test_grpo_config_validation(tmp_path, monkeypatch, override):
    def no_loading(*args, **kwargs):
        raise AssertionError("model loaded before config validation finished")

    monkeypatch.setattr(training.models, "load_policy", no_loading)
    cfg = {
        "stage": "zero",
        "dataset": _write(tmp_path / "d.jsonl", RL_ROWS),
        "output_dir": str(tmp_path / "out"),
        **override,
    }
    with pytest.raises(ValueError):
        training.train_grpo(cfg)
    assert not (tmp_path / "out").exists()


def test_dataset_validated_before_model_loading(tmp_path):
    bad = _write(tmp_path / "bad.jsonl", [{"prompt": 3, "response": "x"}])
    with pytest.raises((ValueError, TypeError)):
        training.train_sft(
            {
                "stage": "cold_start",
                "backend": "hf",
                "model_name_or_path": "no/such-model",
                "dataset": bad,
                "output_dir": str(tmp_path / "out"),
            }
        )


def test_mixed_sft_requires_original_base(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / training.METADATA_FILE).write_text(json.dumps({"stage": "reasoning"}))
    with pytest.raises(ValueError, match="original base"):
        training.train_sft(
            {
                "stage": "mixed_sft",
                "model_name_or_path": str(ckpt),
                "dataset": _write(tmp_path / "d.jsonl", SFT_ROWS),
                "output_dir": str(tmp_path / "o"),
            }
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"stage": "cold_start", "lineage": [{"stage": "base", "source": "m"}]},
        {"stage": "reasoning", "lineage": [{"stage": "base"}, {"stage": "cold_start"}]},
        # A zero checkpoint whose own lineage already includes SFT is not R1-Zero either.
        {"stage": "zero", "lineage": [{"stage": "base"}, {"stage": "cold_start"}]},
    ],
)
def test_zero_stage_rejects_non_base_checkpoints(tmp_path, monkeypatch, metadata):
    def no_loading(*args, **kwargs):
        raise AssertionError("model loaded before the base-model guard")

    monkeypatch.setattr(training.models, "load_policy", no_loading)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / training.METADATA_FILE).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="base model"):
        training.train_grpo(
            {
                "stage": "zero",
                "model_name_or_path": str(ckpt),
                "dataset": _write(tmp_path / "d.jsonl", RL_ROWS),
                "output_dir": str(tmp_path / "o"),
            }
        )
    assert not (tmp_path / "o").exists()


def test_zero_source_guard_accepts_unknown_and_pure_zero_sources():
    training._check_zero_source(None, "some/hub-id")  # cannot be proven; accepted
    training._check_zero_source({"stage": "zero", "lineage": [{"stage": "base", "source": "m"}]}, "ckpt")


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
def test_grpo_uses_frozen_old_and_reference_log_probs(tmp_path, monkeypatch):
    def fake_score(example, response, stage="zero", **hooks):
        # Deterministic, varied rewards so groups are not tied.
        return {"accuracy": 0.0, "total": (sum(map(ord, response)) % 11) / 10.0}

    calls = []
    real_loss = training.grpo_loss

    def spy(log_probs, old_log_probs, ref_log_probs, mask, advantages, **kwargs):
        calls.append(
            {
                "grad": (
                    log_probs.requires_grad,
                    old_log_probs.requires_grad,
                    ref_log_probs.requires_grad,
                    advantages.requires_grad,
                ),
                "lp": log_probs.detach().clone(),
                "old": old_log_probs.clone(),
                "ref": ref_log_probs.clone(),
                "mask": mask.clone(),
                "adv": advantages.clone(),
            }
        )
        return real_loss(log_probs, old_log_probs, ref_log_probs, mask, advantages, **kwargs)

    monkeypatch.setattr(training, "score_response", fake_score)
    monkeypatch.setattr(training, "grpo_loss", spy)
    out = tmp_path / "grpo"
    summary, policy = training._train_grpo(
        {
            "stage": "zero",
            "dataset": _write(tmp_path / "rl.jsonl", RL_ROWS),
            "output_dir": str(out),
            "batch_size": 2,
            "group_size": 4,
            "max_steps": 2,
            "update_epochs": 2,
            "max_new_tokens": 8,
            "learning_rate": 1e-2,
            **LENGTHS,
        }
    )
    assert len(calls) == 4
    first, second, third, fourth = calls
    assert first["grad"] == (True, False, False, False)
    assert first["lp"].shape[:2] == (2, 4)
    assert first["adv"].abs().sum() > 0
    # Old log-probs are frozen across update epochs; first epoch is on-policy.
    torch.testing.assert_close(first["old"], second["old"])
    torch.testing.assert_close(first["lp"], first["old"])
    # Reference is the frozen stage-start policy.
    torch.testing.assert_close(first["ref"], second["ref"])
    torch.testing.assert_close(first["ref"], first["old"])
    assert not torch.allclose(second["lp"] * second["mask"], second["old"] * second["mask"])
    # On a later rollout the old policy refreshes, but the reference stays at
    # the stage initialization rather than following the updated actor.
    torch.testing.assert_close(third["old"], fourth["old"])
    torch.testing.assert_close(third["ref"], fourth["ref"])
    torch.testing.assert_close(third["lp"], third["old"])
    assert not torch.allclose(third["old"], third["ref"])

    assert summary["parameter_delta_l2"] > 0
    assert 0.0 <= summary["truncation_rate"] <= 1.0
    reloaded = models.load_policy("tiny", str(out))
    assert models.parameter_l2_distance(reloaded.model, policy.model) == 0.0
    metadata = json.loads((out / training.METADATA_FILE).read_text())
    assert metadata["kind"] == "grpo" and metadata["config"]["reduction"] == "sequence"


def test_grpo_rejects_general_examples_outside_all_stage(tmp_path):
    rows = RL_ROWS + [{"prompt": "Write a haiku.", "domain": "general"}]
    data = _write(tmp_path / "rl.jsonl", rows)
    with pytest.raises(ValueError, match="general"):
        training.train_grpo({"stage": "reasoning", "dataset": data, "output_dir": str(tmp_path / "o")})
    with pytest.raises(ValueError, match="helpfulness"):
        training.train_grpo({"stage": "all", "dataset": data, "output_dir": str(tmp_path / "o")})


@pytest.mark.parametrize("beta", [0.04, 0.0])
def test_grpo_tied_rewards_with_identical_reference_do_not_move_parameters(tmp_path, monkeypatch, beta):
    # Every group is tied (zero advantage) and on every step the policy equals
    # the reference, so the gradient is exactly zero; with no weight decay the
    # optimizer must leave every parameter untouched.
    monkeypatch.setattr(training, "score_response", lambda *args, **kwargs: {"accuracy": 0.0, "total": 0.0})
    summary, policy = training._train_grpo(
        {
            "stage": "zero",
            "dataset": _write(tmp_path / "rl.jsonl", RL_ROWS),
            "output_dir": str(tmp_path / "grpo"),
            "batch_size": 2,
            "group_size": 4,
            "max_steps": 3,
            "max_new_tokens": 64,
            "learning_rate": 1e-2,
            "beta": beta,
            **LENGTHS,
        }
    )
    assert summary["groups_with_nonzero_advantage"] == 0
    assert summary["parameter_delta_l2"] == 0.0
    fresh = models.load_policy("tiny", None, seed=42)
    assert models.parameter_l2_distance(fresh.model, policy.model) == 0.0


def test_grpo_verifier_only_examples_need_no_answer(tmp_path):
    rows = [
        {"id": "c1", "prompt": "Write a function f.", "tests": ["assert f() == 1"]},
        {"id": "c2", "prompt": "Write a function g.", "tests": ["assert g() == 2"]},
    ]
    seen = []

    def verifier(example, response):
        seen.append(example["id"])
        return float(len(response) % 2)

    data = _write(tmp_path / "rl.jsonl", rows)
    summary = training.train_grpo(
        {
            "stage": "zero",
            "dataset": data,
            "output_dir": str(tmp_path / "grpo"),
            "callbacks": {"verifier": verifier},
            "group_size": 2,
            "max_steps": 1,
            "max_new_tokens": 4,
            **LENGTHS,
        }
    )
    assert summary["steps"] == 1 and len(seen) == 2
    metadata = json.loads((tmp_path / "grpo" / training.METADATA_FILE).read_text())
    assert metadata["config"]["callbacks"]["verifier"].startswith("<callable")

    # Without a verifier the reference answer stays mandatory.
    with pytest.raises(ValueError):
        training.train_grpo({"stage": "zero", "dataset": data, "output_dir": str(tmp_path / "o")})
    # A present answer of the wrong type is rejected even with a verifier.
    bad = _write(tmp_path / "bad.jsonl", [{"prompt": "x", "answer": 3}])
    with pytest.raises(ValueError):
        training.train_grpo(
            {"stage": "zero", "dataset": bad, "output_dir": str(tmp_path / "o"), "callbacks": {"verifier": verifier}}
        )
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# Generation and smoke
# ---------------------------------------------------------------------------
def test_generate_candidates_preserves_records(tmp_path):
    data = _write(tmp_path / "rl.jsonl", RL_ROWS)
    rows = training.generate_candidates({"dataset": data, "num_samples": 4, "max_new_tokens": 8, **LENGTHS})
    assert [r["id"] for r in rows] == ["a", "b"]
    assert all(r["answer"] in {"2", "4"} and len(r["responses"]) == 4 for r in rows)
    with pytest.raises(ValueError):
        training.generate_candidates({"dataset": data, "num_samples": 2})


def test_generate_candidates_accepts_unlabeled_prompts(tmp_path):
    rows = [{"id": "u", "prompt": "What is 5 + 5?"}, {"id": "g", "prompt": "Hi.", "domain": "general"}]
    out = training.generate_candidates(
        {"dataset": _write(tmp_path / "u.jsonl", rows), "num_samples": 4, "max_new_tokens": 4, **LENGTHS}
    )
    assert [r["id"] for r in out] == ["u", "g"]
    assert all("answer" not in r and len(r["responses"]) == 4 for r in out)
    bad = _write(tmp_path / "bad.jsonl", [{"prompt": "x", "answer": 3}])
    with pytest.raises(ValueError):
        training.generate_candidates({"dataset": bad, "num_samples": 4})


def test_smoke_end_to_end(tmp_path):
    out = tmp_path / "smoke"
    summary = training.smoke(str(out))
    for name in ("smoke_summary.json", "candidates.jsonl", "evaluation.json"):
        assert (out / name).is_file()
    assert (out / "cold_start" / models.TINY_WEIGHTS_FILE).is_file()
    assert (out / "zero" / models.TINY_WEIGHTS_FILE).is_file()
    assert summary["reload"]["sft_delta_l2_vs_fresh_init"] > 0
    assert math.isfinite(summary["reload"]["sft_probe_log_prob"])
    assert summary["grpo"]["steps"] == 2
    with pytest.raises(FileExistsError):
        training.smoke(str(out))
