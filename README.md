# r1-grpo

An inspectable, pure-PyTorch **study implementation** of the training pipeline
described in *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning* (DeepSeek-AI, arXiv:2501.12948 **v1**, 2025-01-22,
<https://arxiv.org/html/2501.12948v1>).

It implements:

* **GRPO** with the paper's sequence-likelihood objective (clipped ratio, KL to a
  frozen reference, group-normalised advantages), plus an optional, clearly
  labelled token-level variant.
* **R1-Zero**: pure RL from a base model with rule-based accuracy and format
  rewards.
* **R1**: cold-start SFT → reasoning RL (accuracy + language consistency) →
  rejection sampling + mixing with general data + SFT of a **fresh base model**
  for **2 epochs** → all-scenario RL.
* **Distillation**: SFT only, no RL on the student.
* **Evaluation**: pass@1 averaged over `k` samples per question, and majority
  consensus.

> **This is a method implementation, not a benchmark reproduction.**
> [DeepSeek-V3-Base weights](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)
> are public, but the original training data, reward models and complete
> training recipe are not supplied here. Defaults are explicit substitutes, and no
> numerical result from the paper is claimed. See
> [docs/fidelity.md](docs/fidelity.md).

No dataset framework or RL library is used: data is JSONL, configs are JSON, and
the training loop is plain PyTorch so every tensor can be inspected.

## Quickstart (offline, CPU)

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run r1-grpo smoke --output-dir runs/smoke
uv run pytest
```

`smoke` uses the **tiny offline fixture** (a byte/character tokenizer and a small
GRU — a test fixture, not the DeepSeek-V3 architecture). It runs cold-start SFT,
real on-policy R1-Zero GRPO, sampling, evaluation and checkpoint reloading on
CPU in well under a minute, with no model or dataset downloads after installation. The model is random, so it
usually earns **zero reward**; the smoke run checks plumbing, not learning.

`python -m r1_grpo ...` is equivalent to `r1-grpo ...`. Every command has
`--help`, and `--traceback` shows full tracebacks on errors.

## Running the pipelines (toy scale)

Run from the repository root: all paths in the commands and configs are relative
to the current working directory. Checkpoints are never silently overwritten —
choose a new `output_dir` to re-run a stage.

### R1-Zero

```bash
uv run r1-grpo grpo --config configs/r1_zero.json
```

### DeepSeek-R1 (four stages)

```bash
# Stage 1 — cold-start SFT from the base model
uv run r1-grpo sft --config configs/r1_stage1_cold_start_sft.json

# Stage 2 — reasoning-oriented RL (accuracy + language consistency)
uv run r1-grpo grpo --config configs/r1_stage2_reasoning_rl.json

# Stage 3 — rejection sampling, mixing, and SFT of a FRESH base model for 2 epochs
uv run r1-grpo sample --config configs/r1_stage3_sample.json \
    --output runs/r1-stage3/candidates.jsonl
uv run r1-grpo reject --input runs/r1-stage3/candidates.jsonl \
    --output runs/r1-stage3/reasoning_sft.jsonl
uv run r1-grpo mix --reasoning runs/r1-stage3/reasoning_sft.jsonl \
    --general examples/data/general_sft.jsonl \
    --output runs/r1-stage3/mixed_sft.jsonl
uv run r1-grpo sft --config configs/r1_stage3_mixed_sft.json

# Stage 4 — RL for all scenarios (rule rewards + preference callbacks)
uv run r1-grpo grpo --config configs/r1_stage4_all_rl.json
```

`configs/r1_stage3_mixed_sft.json` deliberately has **no** `model_name_or_path`:
the tiny backend then builds a fresh model from `seed`, i.e. the same base as
stage 1, not the stage-2 checkpoint. With the HF backend, set
`model_name_or_path` to the original base model.

With the untrained tiny model, `reject` will typically accept **nothing**: it
warns and writes an empty file, and `mix` then stops with a clear error because
there is no reasoning data. To exercise the remaining plumbing on toy data only,
you may pass `--reasoning examples/data/cold_start_sft.jsonl` to `mix` instead —
this is a stand-in, not rejection-sampled data. To test rejection and mixing
using explicitly hand-written candidate fixtures:

```bash
uv run r1-grpo reject --input examples/data/candidates.jsonl \
    --output runs/r1-stage3/fixture_reasoning_sft.jsonl
uv run r1-grpo mix --reasoning runs/r1-stage3/fixture_reasoning_sft.jsonl \
    --general examples/data/general_sft.jsonl --output runs/r1-stage3/mixed_sft.jsonl
```

Then run the stage-three SFT and stage-four commands above. Stage 4 uses the **untrained toy callbacks** in
[`examples/callbacks.py`](examples/callbacks.py) for helpfulness and
harmlessness; they are illustrations of the hook signatures, not reward models.

### Distillation (SFT only)

```bash
uv run r1-grpo sft --config configs/distill_sft.json   # uses runs/r1-stage3/mixed_sft.jsonl
```

### Evaluation

```bash
uv run r1-grpo sample --config configs/eval_toy.json --output runs/eval/candidates.jsonl
uv run r1-grpo evaluate --input runs/eval/candidates.jsonl --output runs/eval/report.json
```

`evaluate` reports `pass_at_1` (averaged over each question's samples, then over
questions) and `consensus_accuracy` (normalised-answer majority vote) — not "any
sample correct" — plus per-question details. For external correctness judges,
pass the same `--verifier module:function` used with rejection sampling.
[`configs/hf/eval_paper_v1.json`](configs/hf/eval_paper_v1.json) uses the
paper-v1 sampling settings (temperature 0.6, top-p 0.95, up to 32,768 new tokens,
16 samples per question).

## Optional: Hugging Face backend

```bash
uv sync --extra hf --extra dev
uv run r1-grpo grpo --config configs/hf/r1_zero_qwen2.5_math_1.5b.json
```

`Qwen/Qwen2.5-Math-1.5B` is a **small substitute** base chosen so the recipe can
be tried on one GPU. It is not the paper's base model (DeepSeek-V3-Base); in the
paper it only appears as a distillation student. The HF backend does full-weight
fine-tuning on one device (no distributed training), downloads weights on first
use, and never enables `trust_remote_code`. Full-weight training holds the policy,
frozen reference, optimizer state and activations in memory; the 1.5B example
can require tens of GB, depending on sequence length and batch size.

The optional HF tests create a tiny local transformer and tokenizer, so they
exercise the backend without downloading model weights:

```bash
uv run --extra hf --extra dev pytest
```

## Repository layout

```
src/r1_grpo/   core.py (GRPO math), rewards.py, prompts.py, data.py, rejection.py,
               evaluation.py, models.py, training.py, cli.py
configs/       stage configs (tiny) and configs/hf/ (optional HF backend)
examples/      toy JSONL data and illustrative callbacks
docs/          method.md, fidelity.md, data.md
tests/         offline tests
```

## Documentation

* [docs/method.md](docs/method.md) — objective, rewards, pipelines, evaluation
* [docs/fidelity.md](docs/fidelity.md) — what follows the paper, what is a substitute
* [docs/data.md](docs/data.md) — JSONL schemas, config keys, callbacks

## Citation

If you use this code, please cite the DeepSeek-R1 paper; see
[CITATION.cff](CITATION.cff).

## License

No license has been chosen for this repository yet. The DeepSeek-R1 paper and
any model weights you download are subject to their own terms.
