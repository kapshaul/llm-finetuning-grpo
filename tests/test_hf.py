"""Optional offline tests for the Hugging Face backend.

A tiny randomly initialized GPT-2 and a word-level tokenizer are built and saved
locally, so nothing is downloaded and no remote code is trusted. Skipped when
the optional ``transformers``/``tokenizers`` packages are not installed.
"""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

from r1_grpo import models, training  # noqa: E402
from r1_grpo.prompts import format_prompt  # noqa: E402

SFT_ROWS = [
    {"id": "a", "prompt": "What is 1 + 1?", "response": "<think>1 + 1 = 2</think> <answer>2</answer>"},
    {"id": "b", "prompt": "What is 2 + 2?", "response": "<think>2 + 2 = 4</think> <answer>4</answer>"},
]
RL_ROWS = [
    {"id": "a", "prompt": "What is 1 + 1?", "answer": "2"},
    {"id": "b", "prompt": "What is 2 + 2?", "answer": "4"},
]
LENGTHS = {"max_prompt_tokens": 256, "max_seq_length": 320}
EOS_TOKEN = "<eos>"


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def base_model_dir(tmp_path_factory):
    """Save a tiny random GPT-2 plus a word-level fast tokenizer; EOS doubles as PAD/BOS."""
    texts = [format_prompt(r["prompt"]) for r in SFT_ROWS] + [r["response"] for r in SFT_ROWS]
    words = sorted({word for text in texts for word, _ in Whitespace().pre_tokenize_str(text)})
    vocab = {token: index for index, token in enumerate(["<unk>", EOS_TOKEN, *words])}

    backend = tokenizers.Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
        bos_token=EOS_TOKEN,
    )
    eos_id = vocab[EOS_TOKEN]
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=len(vocab),
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=512,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=eos_id,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    model = transformers.GPT2LMHeadModel(config)
    path = tmp_path_factory.mktemp("hf") / "base"
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_hf_policy_loads_with_eos_as_pad(base_model_dir):
    policy = models.load_policy("hf", base_model_dir)
    assert policy.backend == "hf"
    assert policy.eos_token_id == policy.pad_token_id
    ids = policy.encode_prompt(format_prompt(RL_ROWS[0]["prompt"]))
    assert ids and policy.tokenizer.unk_token_id not in ids


def test_hf_masking_sampling_and_cached_logits_match_full_forward(base_model_dir, monkeypatch):
    policy = models.load_policy("hf", base_model_dir)
    policy.model.eval()
    eos = policy.eos_token_id
    prompt = policy.encode_prompt(format_prompt(RL_ROWS[0]["prompt"]))
    num_samples, budget = 3, 6

    # Record the cached-decoding logits the sampler sees; force row 0 to emit EOS
    # at step 1 and keep the other rows running to the budget.
    seen_logits = []
    real_sampler = models.sample_next_tokens
    filler = policy.encode_text("is")[0]

    def sampler(logits, temperature, top_p, generator):
        tokens = real_sampler(logits, temperature, top_p, generator)
        step = len(seen_logits)
        seen_logits.append(logits.detach().float().cpu().clone())
        tokens = torch.where(tokens == eos, torch.full_like(tokens, filler), tokens)
        if step == 1:
            tokens[0] = eos
        return tokens

    monkeypatch.setattr(models, "sample_next_tokens", sampler)
    samples = models.sample_completions(policy, prompt, num_samples, budget, 1.0, 1.0, torch.Generator().manual_seed(0))
    assert samples[0].token_ids[-1] == eos and len(samples[0].token_ids) == 2
    assert samples[0].stopped_on_eos
    for sample in samples[1:]:
        assert len(sample.token_ids) == budget and not sample.stopped_on_eos
        assert eos not in sample.token_ids

    # Unequal prompts and completions; EOS (== PAD id) must be scored, padding must not.
    other_prompt = policy.encode_prompt(format_prompt(RL_ROWS[1]["prompt"]))[:-3]
    prompts = [prompt] * num_samples + [other_prompt]
    completions = [s.token_ids for s in samples] + [[eos]]
    with torch.no_grad():
        log_probs, mask = models.completion_log_probs(policy, prompts, completions)
    assert mask.tolist()[0] == [1.0, 1.0] + [0.0] * (budget - 2)
    assert mask.tolist()[-1] == [1.0] + [0.0] * (budget - 1)
    assert log_probs[0, 2:].abs().sum().item() == 0.0

    with torch.no_grad():
        for row, (row_prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            ids = torch.tensor([row_prompt + completion])
            logits = policy.model(input_ids=ids[:, :-1]).logits.float()
            logsm = torch.log_softmax(logits, dim=-1)[0]
            expected = torch.stack([logsm[len(row_prompt) - 1 + t, token] for t, token in enumerate(completion)])
            torch.testing.assert_close(log_probs[row, : len(completion)], expected, rtol=1e-4, atol=1e-5)

    # KV-cached next-token logits used for sampling equal the full-forward log-probs.
    for row, sample in enumerate(samples):
        for step, token in enumerate(sample.token_ids):
            cached = torch.log_softmax(seen_logits[step][row], dim=-1)[token]
            torch.testing.assert_close(cached, log_probs[row, step], rtol=1e-4, atol=1e-5)


def test_hf_sft_grpo_reload_and_generate(base_model_dir, tmp_path):
    base = models.load_policy("hf", base_model_dir)
    sft_dir, grpo_dir = tmp_path / "sft", tmp_path / "grpo"

    sft_summary, sft_policy = training._train_sft(
        {
            "stage": "cold_start",
            "backend": "hf",
            "model_name_or_path": base_model_dir,
            "dataset": _write(tmp_path / "sft.jsonl", SFT_ROWS),
            "output_dir": str(sft_dir),
            "epochs": 1,
            "learning_rate": 1e-2,
            **LENGTHS,
        }
    )
    assert sft_summary["backend"] == "hf" and sft_summary["steps"] == 2
    sft_reloaded = models.load_policy("hf", str(sft_dir))
    assert models.parameter_l2_distance(sft_reloaded.model, sft_policy.model) == 0.0
    assert models.parameter_l2_distance(sft_reloaded.model, base.model) > 0
    metadata = json.loads((sft_dir / training.METADATA_FILE).read_text())
    assert metadata["lineage"] == [{"stage": "base", "source": base_model_dir}]

    def verifier(example, response):
        return (sum(map(ord, response)) % 11) / 10.0

    rl_data = _write(tmp_path / "rl.jsonl", RL_ROWS)
    # SFT output is a known non-base checkpoint, so it cannot seed R1-Zero.
    with pytest.raises(ValueError, match="base model"):
        training.train_grpo(
            {
                "stage": "zero",
                "backend": "hf",
                "model_name_or_path": str(sft_dir),
                "dataset": rl_data,
                "output_dir": str(tmp_path / "bad"),
            }
        )

    grpo_summary, grpo_policy = training._train_grpo(
        {
            "stage": "zero",
            "backend": "hf",
            "model_name_or_path": base_model_dir,
            "dataset": rl_data,
            "output_dir": str(grpo_dir),
            "batch_size": 2,
            "group_size": 4,
            "max_steps": 1,
            "max_new_tokens": 6,
            "learning_rate": 1e-2,
            "callbacks": {"verifier": verifier},
            **LENGTHS,
        }
    )
    assert grpo_summary["steps"] == 1
    assert grpo_summary["groups_with_nonzero_advantage"] > 0
    grpo_reloaded = models.load_policy("hf", str(grpo_dir))
    assert models.parameter_l2_distance(grpo_reloaded.model, grpo_policy.model) == 0.0
    assert models.parameter_l2_distance(grpo_reloaded.model, base.model) > 0

    rows = training.generate_candidates(
        {
            "backend": "hf",
            "model_name_or_path": str(grpo_dir),
            "dataset": rl_data,
            "num_samples": 4,
            "max_new_tokens": 4,
            **LENGTHS,
        }
    )
    assert [r["id"] for r in rows] == ["a", "b"]
    assert all(len(r["responses"]) == 4 and all(isinstance(t, str) for t in r["responses"]) for r in rows)
