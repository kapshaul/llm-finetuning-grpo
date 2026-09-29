# Method

This document describes what the code computes. For what is and is not faithful
to the paper, see [fidelity.md](fidelity.md); for file formats, see
[data.md](data.md).

Reference: DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning*, arXiv:2501.12948 **v1** (2025-01-22),
<https://arxiv.org/html/2501.12948v1>.

## 1. GRPO objective (`r1_grpo.core`)

For each prompt `q`, a group of `G` completions `o_1 .. o_G` is sampled from the
frozen rollout policy `pi_old`. Each completion receives a scalar reward `r_i`.

### Advantages — `group_advantages(rewards, eps=1e-8)`

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

Conventions made explicit here (the paper does not specify them):

* `std` is the **population** standard deviation (`correction=0`).
* `eps` guards the division; a group whose rewards are all equal has **zero
  advantage** for every member (it contributes no policy-gradient signal, only
  the KL term).
* Groups are normalised independently; rows of the `[B, G]` tensor never mix.

### Sequence likelihoods (default, `reduction="sequence"`)

The paper's equations are written in terms of sequence probabilities
`pi(o_i | q)`. The default implementation therefore sums masked completion-token
log-probabilities **before** forming the ratio and the KL term:

```
log pi(o_i|q) = sum_t mask_t * log pi(o_{i,t} | q, o_{i,<t})
rho_i         = exp(log pi_theta(o_i|q) - log pi_old(o_i|q))
KL_i          = exp(log pi_ref(o_i|q) - log pi_theta(o_i|q))
                - (log pi_ref(o_i|q) - log pi_theta(o_i|q)) - 1        (>= 0)
J             = mean_i [ min(rho_i * A_i, clip(rho_i, 1-eps_c, 1+eps_c) * A_i) - beta * KL_i ]
loss          = -J   (averaged over prompts in the batch)
```

Sequence-level ratios can become very large or very small for long completions.
The code does **not** silently clamp ratios or rewrite the equation; non-finite
ratios, losses or diagnostics stop training with an error. The KL expression is
evaluated as `expm1(d) - d` to reduce cancellation near zero. Its expectation is
KL under current-policy samples; on reused old-policy rollouts it remains the
paper's sampled penalty, not an exact distribution-level KL measurement.

### Token-level practical variant (`reduction="token"`)

Optional and labelled as a **practical variant**, following the per-token
formulation of GRPO in DeepSeekMath (Shao et al., 2024, arXiv:2402.03300): ratio,
clipping and the KL estimator are computed per token, averaged over each
sequence's completion tokens, then averaged over the group. It is not the
equation printed in R1 v1.

### Frozen quantities

* `old_log_probs` are recorded once per rollout batch and stay fixed through all
  `update_epochs` inner updates.
* `ref_log_probs` come from a reference model frozen at the **start of each RL
  stage** (the stage's initial checkpoint).
* Old log-probs, reference log-probs and advantages are detached; gradients flow
  only through the current policy's log-probs. Padding and prompt positions are
  masked before summation and receive no gradient.
* `beta = 0` skips the KL computation entirely and reports zero KL penalty.
* AdamW uses zero weight decay; no additional parameter penalty is added to GRPO.

### Sampling distribution

The ratio is only an unbiased importance weight if completions were sampled from
exactly the distribution whose log-probs are used. RL rollouts therefore sample
the **raw full-vocabulary softmax** at `temperature = 1.0`, `top_p = 1.0`, with no
top-k, repetition penalty or forced/suppressed tokens. Special tokens stay in the
support. Exact sampled token IDs are kept for training (text is never decoded and
re-tokenised). A completion that hits `max_new_tokens` is truncated **without**
appending EOS.

## 2. Rewards (`r1_grpo.rewards`)

All rewards are rule-based unless supplied by a user callback.

| Stage (`stage`) | Example domain | Reward (summed) |
| --- | --- | --- |
| `zero` (R1-Zero) | reasoning | accuracy + format |
| `reasoning` (R1 stage 2) | reasoning | accuracy + language consistency |
| `all` (R1 stage 4) | reasoning | accuracy + format |
| `all` (R1 stage 4) | general | helpfulness(final answer) + harmlessness(full response) |

* **Accuracy** — extracts the final answer (answer tags, else the last balanced
  `\boxed{...}`, else plain text) and compares it with the reference using safe
  exact / rational comparison. Nothing is `eval`-ed. An answer inside the
  reasoning block is not used when a final summary exists. A custom `verifier`
  callback (e.g. an external, sandboxed code judge) may replace it; generated code
  is never executed in-process.
* **Format** — rewards the reasoning/answer tag structure requested by the prompt.
* **Language consistency** — the share of the reasoning text judged to be in the
  target language (English). The built-in scorer is an illustrative heuristic,
  **not** a linguistic classifier; replace it with a `language_scorer` callback.
* **Helpfulness / harmlessness** — must be provided as callbacks. Training on a
  `general` example without them fails with an error. Helpfulness sees only the
  final answer; harmlessness sees the whole response, reasoning included.
* Zero and reasoning stages refuse `general` examples. R1-Zero uses no neural
  scorers. There are no process reward models, MCTS, critics or length bonuses.

## 3. Prompt (`r1_grpo.prompts.format_prompt`)

A paraphrased instruction asks the model to reason inside think tags and put the
final answer inside answer tags. It is **not** a verbatim copy of the paper's
template and does not require any particular reasoning style (such as
reflection). General-domain records use a direct-answer prompt without requiring
reasoning tags, so simple SFT examples can answer directly. The shared tag-based
reasoning format is an implementation convention: v1 describes special-token
delimiters for R1 cold-start data without publishing their full tokenizer recipe.

## 4. Pipelines

### R1-Zero

Base model -> GRPO (`stage: zero`) with accuracy + format rewards. No SFT.
The trainer refuses known non-Zero training lineages as R1-Zero initialization
and refuses trained package checkpoints for stage-three mixed SFT. Public model IDs without local lineage
metadata still require the caller to select the correct base model.

### R1 (four stages)

1. **Cold-start SFT** (`sft`, `stage: cold_start`) on a small set of readable
   long-CoT examples, starting from the base model.
2. **Reasoning RL** (`grpo`, `stage: reasoning`) from the stage-1 checkpoint with
   accuracy + language-consistency rewards.
3. **Rejection sampling + SFT**:
   * `sample` responses from the stage-2 checkpoint;
   * `reject`: keep only responses that are correct (independent of any format
     bonus), well-structured and readable (no code fences or overlong paragraphs
     in the reasoning, language-consistency threshold), de-duplicated per prompt;
   * `mix` the accepted reasoning rows with general (non-reasoning) SFT data;
   * `sft` (`stage: mixed_sft`) starting again from the **original base model**
     (not the stage-2 checkpoint) for **2 epochs**.
4. **All-scenario RL** (`grpo`, `stage: all`) from the stage-3 checkpoint:
   rule rewards for reasoning prompts, preference callbacks for general prompts.

### Distillation

A student is fine-tuned with **SFT only** (`stage: distill`) on the curated
stage-3 data. No RL is applied to the student.

## 5. Evaluation (`r1_grpo.evaluation`)

For each question, `k` responses are sampled (`temperature 0.6`, `top_p 0.95`,
`k = 16` by default, configurable 4-64; the paper-v1 config uses up to 32,768
new tokens).

* **pass@1** = mean over questions of (fraction of that question's `k` samples
  that are correct). Averaging per question first avoids weighting questions by
  their sample count.
* **Consensus** = majority vote over normalised answers (first occurrence breaks
  ties); invalid or empty answers count as failures.
* "Any sample correct" is **not** reported as pass@1.
