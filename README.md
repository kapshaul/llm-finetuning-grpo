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

## Academic background

*Reading guide:* to run the code, skip to [Quickstart](#quickstart-offline-cpu);
for derivations, implementation conventions and a paper-to-module map, see
[docs/method.md](docs/method.md). Bracketed numbers refer to the
[references](#references) below.

### Research question

Supervised fine-tuning (SFT) teaches a language model to imitate fixed
demonstrations: every token of a reference trajectory is a target. DeepSeek-R1
studies a different learning signal. When the final answer to a task can be
checked by a rule (a number compared with a reference, a program run against
tests), one can sample several attempts, score only their outcomes, and make the
generation strategies that led to better outcomes more probable. The model is
not shown *how* to reason; it is told only how well each complete attempt ended
([1, §2.1–2.2](https://arxiv.org/html/2501.12948v1)).

The paper examines this in two settings. **R1-Zero** applies reinforcement
learning (RL) directly to a pretrained base model, rewarded only for answer
accuracy and output format ([1, §2.2](https://arxiv.org/html/2501.12948v1)).
**R1** responds to the readability problems and language mixing reported for
R1-Zero, and extends training to general tasks, through a multi-stage pipeline
that combines supervised training with RL
([1, §2.3](https://arxiv.org/html/2501.12948v1)).

Three qualifications keep the claim precise. First, "RL without SFT" means
neither "without pretraining" (R1-Zero starts from a large pretrained model) nor
"without supervision": reference answers and verifiers *are* supervision,
delivered at the level of outcomes rather than tokens. Second, an outcome reward
checks the final answer against the verifier. It does not establish that each intermediate
step is valid, nor that the written reasoning faithfully describes the
computation that produced the answer. Third, the behaviours the paper reports
during RL, such as longer responses and reflection on earlier steps, are empirical
observations for one model and data regime, not guaranteed consequences of the
algorithm; and a longer output is not in itself evidence of better reasoning.

### Training pipelines

![R1-Zero, four-stage R1 and SFT-only distillation, with separate weight and data flows](docs/assets/training-pipeline.svg)

*Figure 1. Method structure. Solid arrows carry model checkpoints; dashed
arrows carry data. Stage 3 starts from the original base, and the student starts
from its own checkpoint. Original schematic based on [1, §§2.2–2.4].*

| Pipeline / stage | Initialization | Learning signal | Purpose |
| --- | --- | --- | --- |
| R1-Zero | pretrained base | GRPO: accuracy + format | Test whether outcome rewards alone elicit reasoning |
| R1, 1: cold-start SFT | pretrained base | SFT on a small set of readable long-CoT examples | Readable output format and a stable starting point for RL |
| R1, 2: reasoning RL | stage-1 checkpoint | GRPO: accuracy + language consistency | Improve reasoning while discouraging language mixing |
| R1, 3: rejection sampling + SFT | **fresh original base**, 2 epochs | SFT on filtered stage-2 outputs mixed with general data | Consolidate reasoning together with general abilities |
| R1, 4: all-scenario RL | stage-3 checkpoint | GRPO: rule rewards (reasoning); helpfulness on the final answer + harmlessness on the full response (general) | Improve helpfulness and harmlessness while retaining reasoning |
| Distillation | student's own starting checkpoint | SFT only on curated stage-3 data | Transfer behaviour to a smaller model; no RL on the student |

Distillation here is supervised likelihood training on teacher-generated,
filtered sequences. It does not match the teacher's token distributions or
logits. SFT-only describes the distillation stage, not necessarily the prior
training history of the student checkpoint: the paper also uses an
instruction-tuned starting model ([1, §2.4](https://arxiv.org/html/2501.12948v1)).

In this repository the verifier, language scorer and preference models are
substitute heuristics or user-supplied callbacks, the data are toy fixtures, and
undisclosed training hyperparameters are explicit repository choices; see
[docs/fidelity.md](docs/fidelity.md). Running the pipelines reproduces the
*structure* of the method, not its reported results.

### Group Relative Policy Optimization (GRPO)

![One GRPO group: sampled completions, rewards, normalized advantages and a regularized policy update](docs/assets/grpo-update.svg)

*Figure 2. A worked example of one rollout group. Values illustrate this
repository's normalization convention; they are not experimental results.*

Let $q$ be a prompt and $\pi_\theta$ the autoregressive policy being trained,
$\pi_\theta(o \mid q) = \prod_t \pi_\theta(o_t \mid q, o_{<t})$. A rollout-policy snapshot
$\pi_{\mathrm{old}}$ samples a group of $G$ completions $o_1, \dots, o_G$ for the
same prompt, and each completion receives a scalar reward $r_i$.
In this implementation, the old scores are saved for each rollout batch,
whereas the reference policy $\pi_{\mathrm{ref}}$ is frozen at the start of an RL stage and stays fixed
for the whole stage.

GRPO, introduced in DeepSeekMath
([2, §4.1](https://arxiv.org/html/2402.03300v3)), replaces the learned value
critic of PPO with a baseline computed from the entire group for the *same*
prompt. This repository uses the population standard deviation and a small
stabilizer $\eta = 10^{-8}$:

$$
\bar r = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad
s = \sqrt{\frac{1}{G}\sum_{j=1}^{G} (r_j - \bar r)^2}, \qquad
\hat A_i = \frac{r_i - \bar r}{s + \eta}.
$$

Positive advantages encourage higher completion likelihoods; negative ones
encourage lower likelihoods. These are surrogate incentives: shared model
parameters couple the actual changes across completions. Normalization removes
a common reward offset within a prompt, but difficulty still affects how often
a group contains informative differences. If all $G$
total rewards are equal, the code
sets $\hat A_i = 0$ and the group contributes no policy-gradient signal. For
example, rewards $(0, 1, 1, 2)$ give $\bar r = 1$ and $s = 1/\sqrt 2$, hence
$\hat A \approx (-\sqrt 2, 0, 0, \sqrt 2)$, approximate only because of $\eta$.
Dropping the critic removes a value network, but each prompt still costs $G$
generations plus a forward pass of the reference model.

By default the objective follows the sequence-level equations printed in R1 v1
([1, §2.2.1, Eqs. 1–3](https://arxiv.org/html/2501.12948v1)). For one prompt,
the quantity maximized over $\theta$ is shown below; the code averages it over
the prompts in a batch and minimizes its negative.

$$
\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\mathrm{old}}(o_i \mid q)}.
$$

$$
\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\Big[\min\big(\rho_i \hat A_i,\;
\mathrm{clip}(\rho_i, 1-\epsilon_{\mathrm{clip}}, 1+\epsilon_{\mathrm{clip}})\,\hat A_i\big)
- \beta\, \hat D_i\Big],
$$

$$
\hat D_i = \frac{\pi_{\mathrm{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)}
- \log\frac{\pi_{\mathrm{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1 \;\ge\; 0 .
$$

Clipping removes the incentive to push $\rho_i$ beyond $1+\epsilon_{\mathrm{clip}}$
when $\hat A_i > 0$, or below $1-\epsilon_{\mathrm{clip}}$ when $\hat A_i < 0$. It
discourages large changes within one rollout batch but is not a hard KL bound or
trust-region guarantee. The coefficient $\beta$ weights a penalty that keeps
$\pi_\theta$ near $\pi_{\mathrm{ref}}$; it regularizes the objective and is not
added to the reward, so it never enters $\hat A_i$.

$\rho_i$ is a correct importance weight only if $o_i$ was actually drawn from
$\pi_{\mathrm{old}}$. Training rollouts therefore sample the unmodified softmax
(temperature 1, top-p 1). The evaluation setting (temperature 0.6, top-p 0.95)
defines a different, sharpened and truncated distribution and is used only for
sampling candidates and evaluation, never inside the ratio.

DeepSeekMath's original formulation computes the ratio, clipping and KL term
per token and averages over tokens
([2, §4.1.2](https://arxiv.org/html/2402.03300v3)); it is available as the
labelled `reduction="token"` variant. This repository makes no claim about which
form DeepSeek's own training code used. See
[docs/method.md](docs/method.md#1-grpo-objective-r1_grpocore) for the full
derivation.

### References

1. DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
   via Reinforcement Learning.* arXiv:2501.12948v1.
   <https://arxiv.org/html/2501.12948v1>
2. Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical
   Reasoning in Open Language Models.* arXiv:2402.03300v3.
   <https://arxiv.org/html/2402.03300v3>

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
[CITATION.cff](CITATION.cff). If you use the GRPO objective itself, please also
cite DeepSeekMath [2], where GRPO was introduced.

## License

No license has been chosen for this repository yet. The DeepSeek-R1 paper and
any model weights you download are subject to their own terms.
