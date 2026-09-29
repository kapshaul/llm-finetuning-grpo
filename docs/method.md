# Method

This document is the mathematical companion to the
[README's academic background](../README.md#academic-background). It states
exactly what the code computes, derives the properties that motivate each
choice, and marks which conventions belong to this repository rather than to
the paper. For what is and is not faithful to the paper, see
[fidelity.md](fidelity.md); for file formats, see [data.md](data.md).

References (full entries at the [end](#references)):
[1] DeepSeek-AI, *DeepSeek-R1*, arXiv:2501.12948 **v1** (2025-01-22),
<https://arxiv.org/html/2501.12948v1>;
[2] Shao et al., *DeepSeekMath*, arXiv:2402.03300v3,
<https://arxiv.org/html/2402.03300v3>. GRPO was introduced in [2, §4.1]; R1
applies it and prints a sequence-level form of it [1, §2.2.1].

## 0. Notation

| Symbol | Meaning | In code |
| --- | --- | --- |
| $q$ | a prompt (formatted by `prompts.format_prompt`) | `prompt_ids` |
| $o_i = (o_{i,1}, \dots, o_{i,\lvert o_i\rvert})$ | the $i$-th sampled completion, as token IDs | `flat_completions` |
| $G$ | group size: completions per prompt ($G \ge 2$) | `group_size` |
| $B$ | prompts per RL step | `batch_size` |
| $\pi_\theta$ | the policy being trained | `policy` |
| $\pi_{\mathrm{old}}$ | the policy that generated the current rollouts; fixed during the updates on those rollouts | `old_log_probs` |
| $\pi_{\mathrm{ref}}$ | reference policy, frozen at the start of the RL stage | `reference`, `ref_log_probs` |
| $r_i$ | scalar reward of $o_i$ | `rewards` |
| $\hat A_i$ | group-relative advantage of $o_i$ | `advantages` |
| $m_{i,t} \in \lbrace 0, 1\rbrace$ | completion mask (1 on real completion tokens) | `completion_mask` |
| $\epsilon_{\mathrm{clip}}$ | clipping range of the likelihood ratio | `clip_epsilon` |
| $\beta \ge 0$ | coefficient of the reference penalty | `beta` |
| $\eta$ | stabilizer in the advantage denominator, $10^{-8}$ | `eps` in `group_advantages` |

$\epsilon_{\mathrm{clip}}$ and $\eta$ are unrelated: the first shapes the
objective, the second only prevents division by zero.

Sequence log-likelihoods are sums of masked token log-probabilities:

$$
\log \pi(o_i \mid q) = \sum_{t} m_{i,t}\, \log \pi(o_{i,t} \mid q, o_{i,<t}).
$$

## 1. GRPO objective (`r1_grpo.core`)

For each prompt $q$, a group of $G$ completions $o_1, \dots, o_G$ is sampled from
the frozen rollout policy $\pi_{\mathrm{old}}$. Each completion receives a scalar
reward $r_i$ (Section 2).

### 1.1 Advantages: `group_advantages(rewards, eps=1e-8)`

$$
\bar r = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad
s = \sqrt{\frac{1}{G}\sum_{j=1}^{G} (r_j - \bar r)^2}, \qquad
\hat A_i = \frac{r_i - \bar r}{s + \eta}.
$$

Conventions made explicit here (the paper does not specify them):

* $s$ is the **population** standard deviation (`correction=0`).
* $\eta$ guards the division; a group whose rewards are all exactly equal has
  **zero advantage** for every member, set explicitly rather than left to
  $0/(0+\eta)$. It contributes no policy-gradient signal, only the KL term.
* Groups are normalised independently; rows of the `[B, G]` tensor never mix.

Properties that follow directly from the definition:

* **Zero sum.** $\sum_i \hat A_i = 0$ exactly, because $\sum_i (r_i - \bar r) = 0$.
  This centers the surrogate incentives; shared parameters couple the actual
  probability changes across completions. Floating-point sums may differ
  slightly from zero.
* **Unit scale.** $\frac{1}{G}\sum_i \hat A_i^2 = s^2/(s+\eta)^2 \approx 1$ for
  any non-tied group.
* **Shift and scale invariance.** For $c > 0$ and any $b$, replacing every $r_j$
  by $c\,r_j + b$ leaves $\hat A$ unchanged up to the effect of $\eta$. Two
  consequences matter in practice. First, reward weights act only through
  differences *within* a group; a prompt whose completions differ only in format
  receives advantages of full magnitude driven by format alone. Second,
  normalisation amplifies small differences: with the continuous language
  score of stage `reasoning`, a group of two completions whose rewards differ
  by $10^{-3}$ receives advantages $\approx \pm 1$, exactly as a group whose
  rewards differ by 1 would.
* **Worked example.** Rewards $(0, 1, 1, 2)$ give $\bar r = 1$,
  $s = \sqrt{(1 + 0 + 0 + 1)/4} = 1/\sqrt 2$, and
  $\hat A \approx (-\sqrt 2, 0, 0, \sqrt 2) \approx (-1.414, 0, 0, 1.414)$.

![Illustrative GRPO group with rewards 0, 1, 1, 2 and normalized advantages](assets/grpo-update.svg)

*Original worked example. The KL penalty is separate from the reward and does
not enter the group normalization.*

### 1.2 Sequence objective (default, `reduction="sequence"`)

The paper's equations are written in terms of sequence probabilities
$\pi(o_i \mid q)$ [1, §2.2.1, Eqs. 1–3]. The default implementation therefore
sums masked completion-token log-probabilities **before** forming the ratio and
the KL term. With

$$
\rho_i = \exp\big(\log \pi_\theta(o_i \mid q) - \log \pi_{\mathrm{old}}(o_i \mid q)\big),
\qquad
d_i = \log \pi_{\mathrm{ref}}(o_i \mid q) - \log \pi_\theta(o_i \mid q),
$$

$$
\hat D_i = e^{d_i} - d_i - 1
= \frac{\pi_{\mathrm{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)}
- \log \frac{\pi_{\mathrm{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1,
$$

the loss for a batch of $B$ prompts is

$$
\mathcal{L}_{\mathrm{seq}}(\theta) = -\frac{1}{BG}\sum_{b=1}^{B}\sum_{i=1}^{G}
\Big[\min\big(\rho_{b,i}\hat A_{b,i},\;
\mathrm{clip}(\rho_{b,i}, 1-\epsilon_{\mathrm{clip}}, 1+\epsilon_{\mathrm{clip}})\,\hat A_{b,i}\big)
- \beta\,\hat D_{b,i}\Big].
$$

Because every group has the same size $G$, the mean over all $BG$ completions
equals the mean over prompts of each group's mean.

Sequence-level ratios can become very large or very small for long completions.
The code does **not** silently clamp ratios or rewrite the equation; non-finite
ratios, losses or diagnostics stop training with an error.

### 1.3 Interpretation: a group-baselined policy gradient

Consider the first update on a fresh rollout batch. The rollout policy and the
current policy then have the same parameters, so $\rho_i = 1$ in value. Clipping
is inactive at $\rho_i = 1$, and since $\nabla_\theta \rho_i = \rho_i \nabla_\theta
\log \pi_\theta(o_i \mid q)$, the gradient of the surrogate is

$$
\nabla_\theta\,\frac{1}{G}\sum_i \min(\cdot) \;=\; \frac{1}{G}\sum_i \hat A_i\,
\nabla_\theta \log \pi_\theta(o_i \mid q).
$$

This has the form of a REINFORCE-style estimator with a group baseline and
group scaling. Both statistics depend on the sampled completions, including
the completion being scored; it should not be identified with an unbiased
estimator of the unnormalized expected-reward gradient.

Although the mean advantage and the initial surrogate value are zero, its
gradient need not be: the vectors $\nabla_\theta \log \pi_\theta(o_i \mid q)$
differ across completions. A zero loss value alone does not establish that no
learning occurred. With the default `update_epochs = 1` every update is
of this kind, so clipping never binds. It becomes active only when
`update_epochs > 1`: in later epochs $\rho_i \ne 1$, and the $\min$ removes the
incentive to raise $\rho_i$ above $1+\epsilon_{\mathrm{clip}}$ when
$\hat A_i > 0$, or lower it below $1-\epsilon_{\mathrm{clip}}$ when
$\hat A_i < 0$. This reduces the incentive for large policy changes on reused
data but is not a hard bound on the KL divergence or a trust-region guarantee.
The loss contains no constraint, and gradient steps driven by completions that
are not clipped can still move the policy further than the clip range suggests.

The penalty's gradient is, using $\partial \hat D_i / \partial \log
\pi_\theta(o_i \mid q) = 1 - e^{d_i}$,

$$
\nabla_\theta \hat D_i = \Big(1 - \frac{\pi_{\mathrm{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)}\Big)
\nabla_\theta \log \pi_\theta(o_i \mid q).
$$

It vanishes when $\pi_\theta = \pi_{\mathrm{ref}}$ on the sample, so at the
start of a stage the penalty exerts no force. Once the policy drifts, it lowers
the likelihood of completions that $\pi_\theta$ now favours more than
$\pi_{\mathrm{ref}}$ does, and raises those it favours less. $\beta$ weights
this regulariser in the objective; it is not part of $r_i$ and does not enter
$\hat A_i$.

### 1.4 The KL estimator

**Non-negativity.** Because $e^x$ is convex, it lies above its tangent at 0:
$e^{d} \ge 1 + d$ for all real $d$, with equality only at $d = 0$. Hence
$\hat D_i \ge 0$ for every sample, unlike the naive estimator $-d_i$, which can
be negative. The code evaluates $\hat D_i$ as `expm1(d) - d`, which is accurate
near $d = 0$ and avoids the cancellation in a direct `exp(d) - 1`.

**Unbiasedness, and its conditions.** Let $o \sim \pi_\theta(\cdot \mid q)$,
where the distribution is the one induced over completions by the sampler
(including the token budget). Then

$$
\mathbb{E}_{o\sim\pi_\theta}\!\left[e^{d}\right]
= \sum_{o:\,\pi_\theta(o\mid q)>0} \pi_{\mathrm{ref}}(o \mid q),
\qquad
\mathbb{E}_{o\sim\pi_\theta}\!\left[-d\right]
= \mathrm{KL}\big(\pi_\theta(\cdot\mid q)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid q)\big),
$$

so $\mathbb{E}[\hat D] = \mathrm{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})$
exactly when every completion with $\pi_{\mathrm{ref}}(o \mid q) > 0$ also has
$\pi_\theta(o \mid q) > 0$ (compatible support). Two conditions are therefore
required: the samples come from the **current** policy, and the supports are
compatible. On the first update epoch of a rollout both hold for the
full-vocabulary sampler used here. On later epochs the samples come from
$\pi_{\mathrm{old}} \ne \pi_\theta$ and $\hat D$ is a non-negative sampled
penalty, as in the paper's formulation, but not an unbiased estimate of the
distribution-level KL.

These statements concern the *value* of the estimator. The code differentiates
$\hat D_i$ with the samples held fixed, and no claim is made that the resulting
gradient is an unbiased estimate of $\nabla_\theta \mathrm{KL}$.

### 1.5 Token-level practical variant (`reduction="token"`)

This option is labelled as a **practical variant**. It follows the per-token,
outcome-supervised formulation of GRPO in DeepSeekMath [2, §4.1.2]: ratio,
clipping and KL are computed per token, and the sample's scalar advantage is
broadcast to all of its completion tokens. With
$\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\mathrm{old}}(o_{i,t} \mid q, o_{i,<t})$,
$d_{i,t}$ defined analogously, $\hat D_{i,t} = e^{d_{i,t}} - d_{i,t} - 1$ and
$\lvert o_i\rvert = \sum_t m_{i,t}$:

$$
\mathcal{L}_{\mathrm{tok}}(\theta) = -\frac{1}{BG}\sum_{b,i}\,\frac{1}{\lvert o_i\rvert}
\sum_{t} m_{i,t}\Big[\min\big(\rho_{i,t}\hat A_i,\;
\mathrm{clip}(\rho_{i,t}, 1-\epsilon_{\mathrm{clip}}, 1+\epsilon_{\mathrm{clip}})\,\hat A_i\big)
- \beta\,\hat D_{i,t}\Big].
$$

Tokens are averaged **within each completion first, then across completions**.
This is not a global token-weighted mean over the batch. Every completion has
equal total weight, so each token of a long completion carries less weight than
each token of a short one. The per-token KL term estimates, per visited context,
$\mathrm{KL}(\pi_\theta(\cdot \mid q, o_{<t}) \Vert \pi_{\mathrm{ref}}(\cdot \mid q, o_{<t}))$
under the same current-policy and support conditions as above. Its
length-normalised average is not the sequence KL, which is a sum over
positions. Computing each ratio separately avoids multiplying ratios over a long
completion, but neither clipping nor this reduction guarantees numerical
stability. This variant is not the equation printed in R1 v1.

### 1.6 Masks, EOS and truncation

* $m_{i,t} = 1$ exactly on the sampled completion tokens. A **sampled** EOS is
  part of the completion (it is the action that ended the sequence), so its
  log-probability is trained.
* A completion that reaches the token budget (`max_new_tokens`, capped by
  `max_seq_length`) is truncated **without** appending EOS. No artificial EOS
  ever enters the mask, so the model is never credited or penalised for a stop
  action it did not take.
* Prompt and padding positions are masked before summation and receive no
  gradient. Rewards are computed on the decoded text with EOS removed.

### 1.7 Frozen quantities

* `old_log_probs` are recorded once per rollout batch and stay fixed through all
  `update_epochs` inner updates. $\pi_{\mathrm{old}}$ is therefore refreshed at
  every RL step.
* `ref_log_probs` come from a reference model frozen at the **start of each RL
  stage** (the stage's initial checkpoint) and never refreshed within the stage.
  Before the first nonzero-gradient update, the actor and the
  reference are identical, and the code reuses the rollout log-probabilities as
  reference scores. This does not change any value: it only avoids spurious
  rounding differences between two forward passes of identical weights.
* Old log-probs, reference log-probs and advantages are detached; gradients flow
  only through the current policy's log-probs.
* `beta = 0` skips the KL computation entirely (no reference model is built)
  and reports zero KL penalty.
* AdamW uses zero weight decay; no additional parameter penalty is added to GRPO.
  Before any nonzero gradient has occurred, tied groups and a policy identical
  to the reference leave parameters unchanged. After learning begins, AdamW
  momentum from previous gradients can move parameters even on a later
  zero-gradient step.
* There is no value critic, process reward model or tree search (MCTS).

### 1.8 Sampling distribution

The ratio is only a valid importance weight if completions were sampled from
exactly the distribution whose log-probs are used. At each prefix, temperature $\tau$ transforms the next-token probabilities
in proportion to $\pi_{\mathrm{old}}(\cdot \mid q,o_{<t})^{1/\tau}$;
nucleus sampling then restricts their support and renormalizes. These operations
change the distribution of completed sequences, so the unmodified model
likelihood would be the wrong denominator for $\rho_i$. No behaviour-policy correction is
implemented. RL rollouts therefore sample the **raw full-vocabulary softmax** at
`temperature = 1.0`, `top_p = 1.0`, with no top-k, repetition penalty or
forced/suppressed tokens, and the trainer rejects any other setting. Special
tokens stay in the support. Exact sampled token IDs are kept for training (text
is never decoded and re-tokenised). Evaluation and rejection-sampling
candidates use a separate sampler configuration (Section 5).

## 2. Rewards (`r1_grpo.rewards`)

All rewards are rule-based unless supplied by a user callback. Routing in
`score_response`, with plain unweighted sums (a repository default; the paper
does not publish weights):

| Stage (`stage`) | Example domain | Reward $r$ (summed) |
| --- | --- | --- |
| `zero` (R1-Zero) | reasoning | accuracy + format |
| `reasoning` (R1 stage 2) | reasoning | accuracy + language consistency |
| `all` (R1 stage 4) | reasoning | accuracy + format |
| `all` (R1 stage 4) | general | helpfulness(prompt, final answer) + harmlessness(prompt, full response) |

Note that stage `reasoning` has no format term. With the built-in scorers,
accuracy and format are binary, so a stage-`zero` reward takes only the values
$\lbrace 0, 1, 2\rbrace$ and ties within a group are common.

Paper sources: accuracy and format rewards for R1-Zero [1, §2.2.2]; language
consistency as the proportion of target-language words in the chain of thought,
summed with accuracy [1, §2.3.2]; helpfulness judged on the final summary and
harmlessness on the whole response [1, §2.3.4].

* **Accuracy**: extracts the final answer (answer tags, else the last balanced
  `\boxed{...}`, else plain text) and compares it with the reference using safe
  exact / rational comparison. Nothing is `eval`-ed. An answer inside the
  reasoning block is not used when a final summary exists, and unterminated
  reasoning yields no answer. A custom `verifier` callback (e.g. an external,
  sandboxed code judge) may replace it; generated code is never executed
  in-process.
* **Format**: rewards the reasoning/answer tag structure requested by the
  prompt. It is strict and binary: 1 only if the response is exactly one think
  block followed by one answer block, both non-blank, with nothing but
  whitespace around them.
* **Language consistency**: the share of the reasoning text judged to be in the
  target language (English). The built-in scorer is an illustrative heuristic,
  **not** a linguistic classifier; replace it with a `language_scorer` callback.
* **Helpfulness / harmlessness**: must be provided as callbacks. Training on a
  `general` example without them fails with an error. Helpfulness sees only the
  final answer; harmlessness sees the whole response, reasoning included.
* Zero and reasoning stages refuse `general` examples. R1-Zero uses no neural
  scorers. There are no process reward models, MCTS, critics or length bonuses.

The following are **implementation conventions** of this repository, not
details disclosed by the paper:

* the language heuristic counts a word as English when it is written in ASCII
  letters, counts each CJK/kana/hangul character as a separate non-English
  word, scores letter-free (e.g. purely mathematical) reasoning as 1 and blank
  reasoning as 0;
* the strict format rule above, including "each tag exactly once";
* answer normalisation (math delimiters, `\text{}` wrappers, thousands
  separators, degree signs) and exact rational equivalence, so that `0.5`,
  `1/2`, `\frac{1}{2}` and `50%` agree;
* malformed answer tags (unclosed, stray or nested) yield no answer.

## 3. Prompt (`r1_grpo.prompts.format_prompt`)

A paraphrased instruction asks the model to reason inside think tags and put the
final answer inside answer tags. It is **not** a verbatim copy of the paper's
template [1, §2.2.3] and does not require any particular reasoning style (such
as reflection). General-domain records use a direct-answer prompt without
requiring reasoning tags, so simple SFT examples can answer directly. The shared
tag-based reasoning format is an implementation convention: v1 describes
special-token delimiters for R1 cold-start data without publishing their full
tokenizer recipe.

## 4. Pipelines

![Training stages and the distinction between weight initialization and data flow](assets/training-pipeline.svg)

*Original method schematic. The fresh-base restart and SFT-only student are
explicit; arrows do not imply benchmark reproduction.*

### Supervised objective (`training.train_sft`)

All SFT stages minimise completion-only cross-entropy. For a minibatch of
prompt/response pairs $(x_n, y_n)$, where $y_n$ is the response tokens followed
by EOS and $m_{n,t}$ masks retained completion positions,

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\,\frac{\sum_{n}\sum_{t} m_{n,t}\,
\log \pi_\theta(y_{n,t} \mid x_n, y_{n,<t})}{\sum_{n}\sum_{t} m_{n,t}}.
$$

Prompt and padding tokens are masked. If a pair exceeds `max_seq_length`, the
completion is cut to fit and the EOS is dropped with it; EOS is supervised only
when retained. The average is over all retained completion tokens in the
minibatch, unlike the per-completion averaging of Section 1.5; with one SFT example per minibatch, the averaging conventions coincide.
These statements concern weighting only; the SFT and GRPO objectives differ.

### R1-Zero

Base model -> GRPO (`stage: zero`) with accuracy + format rewards. No SFT.
The trainer refuses known non-Zero training lineages as R1-Zero initialization
and refuses trained package checkpoints for stage-three mixed SFT. Public model IDs without local lineage
metadata still require the caller to select the correct base model.

### R1 (four stages)

1. **Cold-start SFT** (`sft`, `stage: cold_start`) on a small set of readable
   long-CoT examples, starting from the base model [1, §2.3.1]. The paper gives
   no epoch count for this stage; the configs use 2 as a repository choice.
2. **Reasoning RL** (`grpo`, `stage: reasoning`) from the stage-1 checkpoint with
   accuracy + language-consistency rewards [1, §2.3.2].
3. **Rejection sampling + SFT** [1, §2.3.3]:
   * `sample` responses from the stage-2 checkpoint;
   * `reject`: keep only responses that are correct (independent of any format
     bonus), well-structured and readable (no code fences or overlong paragraphs
     in the reasoning, language-consistency threshold), de-duplicated per prompt;
   * `mix` the accepted reasoning rows with general (non-reasoning) SFT data;
   * `sft` (`stage: mixed_sft`) starting again from the **original base model**
     (not the stage-2 checkpoint) for **2 epochs**, the count stated in the
     paper.

   Rejection sampling is an outcome filter: it retains only completions that the
   verifier accepts and the readability rules pass, so the SFT target is
   the empirical collection of accepted responses. Correctness, readability,
   de-duplication and dataset mixing all shape that distribution.
   Restarting from the base means the final model inherits stage 2 only through
   this data, not through its weights.
4. **All-scenario RL** (`grpo`, `stage: all`) from the stage-3 checkpoint:
   rule rewards for reasoning prompts, preference callbacks for general prompts
   [1, §2.3.4]. The reference policy for this stage is the stage-3 checkpoint.

### Distillation

A student is fine-tuned with **SFT only** (`stage: distill`) on the curated
stage-3 data [1, §2.4]. No RL stage is added during distillation. The student
starts from its own selected checkpoint; this need not be an untouched
pretraining checkpoint, since the paper includes Llama-3.3-70B-Instruct among
its starting models. "SFT only" therefore describes this procedure, not every
stage in the student's earlier training history. The tiny config builds a
fresh fixture from `seed`; with the HF backend, the caller selects the student's
starting checkpoint. Unlike `mixed_sft`, this stage has no fresh-base lineage
restriction.

This is *sequence-level* distillation: minimising $\mathcal{L}_{\mathrm{SFT}}$
on teacher-generated, filtered sequences fits the student to the empirical
distribution of those sequences. It differs from logit (token-distribution)
distillation, which would minimise a divergence such as
$\sum_t \mathrm{KL}\big(p_{\mathrm{teacher}}(\cdot \mid x, y_{<t}) \,\Vert\, \pi_\theta(\cdot \mid x, y_{<t})\big)$
and requires the teacher's per-token distributions and a shared vocabulary.
Logit distillation is not implemented.

## 5. Evaluation (`r1_grpo.evaluation`)

For each question, `k` responses are sampled (`temperature 0.6`, `top_p 0.95`,
`k = 16` by default, configurable 4-64; the paper-v1 config uses up to 32,768
new tokens) [1, §3]. This sampler is separate from the RL sampler and is also
used to generate rejection-sampling candidates.

Let $c_{q,j} \in \lbrace 0, 1\rbrace$ indicate that sample $j$ of question $q$ is
correct (rule-based, or `verifier(...) >= 1`), with $k_q$ samples for question
$q$ and $N$ questions.

* **pass@1** is the mean over questions of the fraction of that question's
  samples that are correct:

  $$
  \text{pass@1} = \frac{1}{N}\sum_{q=1}^{N}\frac{1}{k_q}\sum_{j=1}^{k_q} c_{q,j}.
  $$

  Averaging per question first avoids weighting questions by their sample
  count. For each question the inner mean is an unbiased estimate of the
  probability that a single sample is correct.
* **Consensus** is a majority vote over canonical answer keys (normalised
  strings, with exact rationals reduced so that `0.5` and `1/2` vote together);
  the first occurrence breaks ties. The question counts as correct if the
  winning key is non-empty and the first sample carrying that key is correct.
  Invalid or empty answers form one bucket and count as failures if they win.
* "Any sample correct" (pass@k) is **not** reported as pass@1.

The report also contains `format_rate` and `invalid_answer_rate`, and
per-question details (extracted answers, per-sample correctness).

## 6. Observable diagnostics and their limits

Each GRPO checkpoint's `metrics.json` records, per step: `reward_mean`,
`reward_std`, `reward_components`, `groups_with_nonzero_advantage`,
`completion_tokens_mean`, `truncation_rate`; and per update: `loss`,
`grad_norm`, `surrogate`, `kl`, `clip_fraction`, `ratio_mean/min/max` and
`advantage_mean`. The summary adds `parameter_delta_l2` (tiny backend only).
How to read them:

* `groups_with_nonzero_advantage` counts groups whose rewards were not all
  equal. Only these groups produce a policy gradient. If it is zero, the step
  has no new surrogate gradient; a KL gradient or existing optimizer momentum
  can still move parameters.
* `reward_std` is the population standard deviation over **all** $BG$ rewards
  of the step. It mixes between-prompt and within-prompt variation and does not
  directly measure the within-group spread that GRPO learns from.
* `advantage_mean` is approximately zero by construction (Section 1.1) and
  serves only as a sanity check.
* `clip_fraction` is the fraction of completions (or tokens) at which clipping
  binds; with `update_epochs = 1` it is expected to be zero (Section 1.3).
* `kl` is the mean of $\hat D$ (Section 1.4; per-completion token averages
  for the token variant). It is a sampled penalty, not an exact divergence, and
  it is zero at the start of every stage.
* `truncation_rate` is the fraction of rollouts that hit the token budget.
  Truncated completions often contain no extractable answer (an unterminated
  think block yields none), so accuracy and
  length must be read together: rising `completion_tokens_mean` may reflect
  more deliberation or simply more truncation, and in neither case is it
  evidence of better reasoning by itself.

Checkpoint metadata records the dataset path and its SHA-256, the resolved
configuration (including `seed`), library versions and the stage lineage.

## 7. Reproducibility requirements

A result produced with this code is interpretable only if the following are
reported together. This is a statement of requirements, not a record of
experiments; the repository ships no trained results.

* **Provenance**: base model identity, the lineage in each checkpoint, dataset
  hashes, the full configuration, and library versions.
* **Variance**: several seeds per configuration, with dispersion across seeds.
  RL on small groups is noisy. Seeding fixes the CPU sampling generator and the
  data order, but bitwise identity across hardware or accelerator kernels is not
  guaranteed.
* **Evaluation protocol**: held-out questions disjoint from training prompts;
  the sampler settings, $k$ and the token budget; pass@1 as defined above rather
  than pass@k; and the verifier used.
* **Learning signal**: the share of groups with non-zero advantage, KL and
  truncation over training. These show whether the reward actually
  discriminated between completions and whether gains coincide with length
  changes.
* **Scope**: which components were substitutes (see [fidelity.md](fidelity.md)).
  The tiny GRU fixture only exercises the plumbing, and the optional HF backend
  is single-device full-weight fine-tuning of a small substitute base. Neither
  recreates the paper's 671B-parameter base or its benchmark results.

## 8. Paper-to-module map

| Paper element | Source | Module |
| --- | --- | --- |
| Group-relative advantage, clipped surrogate, KL penalty | [1, §2.2.1, Eqs. 1–3]; GRPO introduced in [2, §4.1] | `core.group_advantages`, `core.grpo_loss(reduction="sequence")` |
| Per-token outcome-supervised GRPO | [2, §4.1.2] | `core.grpo_loss(reduction="token")` |
| Rule-based accuracy and format rewards | [1, §2.2.2] | `rewards.correctness_reward`, `rewards.format_reward`, `rewards.score_response` |
| Training template | [1, §2.2.3] | `prompts.format_prompt` (paraphrased) |
| Cold start | [1, §2.3.1] | `training.train_sft` (`cold_start`) |
| Language-consistency reward | [1, §2.3.2] | `rewards.english_word_fraction` or a `language_scorer` callback |
| Rejection sampling and mixed SFT | [1, §2.3.3] | `training.generate_candidates`, `rejection.filter_candidates`, `cli` `mix`, `training.train_sft` (`mixed_sft`) |
| RL for all scenarios | [1, §2.3.4] | `rewards.score_response(stage="all")` with callbacks |
| Distillation | [1, §2.4] | `training.train_sft` (`distill`) |
| Evaluation protocol | [1, §3] | `training.generate_candidates`, `evaluation.evaluate_records` |
| PRM and MCTS (reported as unsuccessful) | [1, §4.2] | not implemented |

## References

1. DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
   via Reinforcement Learning.* arXiv:2501.12948v1.
   <https://arxiv.org/html/2501.12948v1>
2. Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of
   Mathematical Reasoning in Open Language Models.* arXiv:2402.03300v3.
   <https://arxiv.org/html/2402.03300v3>
