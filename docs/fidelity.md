# Fidelity to DeepSeek-R1 v1

This repository is a **study implementation**. It reproduces the *structure* of
the DeepSeek-R1 v1 training recipe (arXiv:2501.12948v1, 2025-01-22) at toy scale.
It does **not** reproduce the paper's benchmark results. The
[DeepSeek-V3-Base weights](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)
are public; the original post-training datasets, preference models and complete
training recipe are not provided here. Smaller backends make the method inspectable.

## Stated in the paper and implemented

| Paper (v1) | Here |
| --- | --- |
| GRPO: group of outputs from the old policy, clipped ratio, KL to a reference policy, group-normalised advantages (Eqs. 1-3) | `core.grpo_loss`, `core.group_advantages`; sequence-likelihood form by default |
| Sampled KL penalty `pi_ref/pi - log(pi_ref/pi) - 1` | Same expression, evaluated with `expm1` for stability; unbiased for KL under current-policy sampling, not generally under reused old-policy rollouts |
| R1-Zero: RL directly on the base model with rule-based accuracy and format rewards, no neural reward model | `stage: zero` |
| Template asking for reasoning then answer inside tags, without content-specific bias | Paraphrased in `prompts.format_prompt` (not copied) |
| Cold-start SFT on readable long-CoT data before RL | `stage: cold_start` |
| Language-consistency reward (share of target-language words in the CoT) summed with accuracy | `stage: reasoning`; heuristic scorer, replaceable |
| Rejection sampling from the RL checkpoint, keep only correct responses; filter mixed language, long paragraphs and code blocks | `sample` + `reject` |
| Combine reasoning data with non-reasoning data | `mix` |
| Fine-tune the **base** model for **two epochs** on the combined data | `stage: mixed_sft`, fresh base, `epochs: 2` |
| Second RL stage over all scenarios: rule rewards for reasoning, reward models for general data; helpfulness judged on the final summary, harmlessness on the whole response | `stage: all` with callbacks |
| Distillation: SFT only, no RL on students | `stage: distill` |
| Evaluation: sampling with temperature 0.6, top-p 0.95, max 32,768 generated tokens, `k` samples per question, pass@1 averaged over samples; majority-vote consensus | `configs/hf/eval_paper_v1.json`, `evaluation.evaluate_records` |

## Implementation defaults and substitutions

Every value below is **our choice**, not the paper's.

| Item | Default here | Note |
| --- | --- | --- |
| Base model | tiny offline fixture; optional `Qwen/Qwen2.5-Math-1.5B` | The paper's base is DeepSeek-V3-Base. Qwen2.5-Math-1.5B is a **small substitute**; in the paper it appears only as a distillation student. |
| Optimizer, learning rate, schedule | AdamW (`training.py`), gradient clipping `max_grad_norm` 1.0; `learning_rate` 1e-4 (tiny) / 1e-6 (HF RL) / 1e-5 (HF SFT) | Illustrative only. |
| Group size `G` | 4 (tiny) / 8 (HF) | Illustrative. |
| `clip_epsilon`, `beta` | 0.2, 0.04 | Common public GRPO defaults (0.04 is the KL coefficient reported in DeepSeekMath); R1 v1 does not report its values. |
| Batch size, RL steps, `update_epochs` | 1, 2 (tiny), 1 | Toy scale. |
| Reward weights | Plain unweighted sums | The paper says rewards are summed directly for the language reward; other weights are unspecified. |
| RL sampling temperature | 1.0, top-p 1.0 | Chosen so that sampled completions match the log-probs used in the ratio. |
| Advantage std convention | population std, zero advantage for tied groups | Explicit convention. |
| Ratio / KL granularity | sequence (default) or token (practical variant from DeepSeekMath) | See [method.md](method.md). |
| Language classifier | ASCII/English heuristic | Illustrative, **not** a linguistic classifier. |
| Helpfulness / harmlessness reward models | none built in; user callbacks required | `examples/callbacks.py` holds untrained toy illustrations only. |
| Generative reward (DeepSeek-V3 as judge) for some stage-3 data | not implemented | Supply a `verifier` callback if needed. |
| Code verification | callback only | Generated code is never executed in-process. |
| Cold-start epochs | 2 (same as the documented stage-3 count) | Paper gives no count for stage 1. |
| Training data | tiny hand-written toy JSONL files | Synthetic fixtures, never presented as the paper's thousands of cold-start examples or ~600k reasoning + ~200k general SFT samples. Supply real curated data for training. |
| Tokenizer / architecture (tiny) | byte/character tokenizer + small GRU | A CPU test fixture, not the DeepSeek-V3 architecture. |

## Out of scope

* Process reward models, MCTS, value/critic networks, length bonuses — none
  (the paper reports PRM and MCTS as unsuccessful attempts).
* Distributed or large-scale training. The HF backend does plain full-weight
  fine-tuning on a single device.
* The DeepSeek-V3 non-reasoning SFT data and the "helpful/harmless" preference
  pipelines.

## What the toy runs demonstrate

The tiny fixture model is randomly initialised. On the toy data it will usually
earn **zero reward**, so the GRPO advantages are often zero and rejection
sampling usually accepts nothing. The toy runs demonstrate that gradients,
masking, checkpointing and the stage plumbing work; they **do not** show learned
reasoning.
