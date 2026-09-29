# Data formats and configs

All datasets are UTF-8 **JSONL** (one JSON object per line). All paths — on the
command line and inside configs — are resolved **relative to the current working
directory**; run commands from the repository root.

The files in `examples/data/` are tiny hand-written toy examples for exercising
the code. They are not the paper's data and are far too small to train anything.

## RL prompts (`grpo`, `sample`)

```json
{"id": "q1", "prompt": "What is 7 + 5?", "answer": "12"}
{"id": "g1", "domain": "general", "prompt": "Give one tip for staying focused."}
```

* `prompt` (string, required) — the raw question. The trainer wraps it with
  `r1_grpo.prompts.format_prompt`; do not pre-format it.
* `answer` (string) — required for reasoning training with the built-in verifier.
  It may be omitted when a custom verifier uses other metadata, or when sampling
  responses without scoring them. A provided answer must still be a string.
* `domain` — `"reasoning"` (default) or `"general"`. General rows are only allowed
  in `stage: all` and need helpfulness and harmlessness callbacks.
* `id` and other extra fields are optional and preserved where practical.

## SFT rows (`sft`)

```json
{"id": "c1", "domain": "reasoning", "prompt": "What is 7 + 5?", "response": "<think>\n...\n</think>\n<answer>\n\\boxed{12}\n</answer>"}
```

* `prompt`, `response` (non-empty strings, required). The loss covers the
  response tokens and the EOS token only; prompt and padding are masked.
  If the response exceeds the sequence limit, only retained tokens are
  supervised; no artificial EOS is appended at the truncation boundary.
* Set `domain: "general"` for direct-answer examples. Their prompt does not
  require a `<think>` section.

## Candidates (`sample` output; `reject` and `evaluate` input)

```json
{"id": "q1", "prompt": "What is 7 + 5?", "answer": "12", "responses": ["...", "..."]}
```

`reject` writes SFT rows `{prompt, response, domain: "reasoning", sample_index,
...metadata}`; `answer` and `responses` are dropped.
An empty output is valid (typical for an untrained model), but `mix` and `sft`
refuse empty data with a clear error. `mix` tags every row with `domain` and a
`source` provenance field (`"reasoning"` / `"general"`, unless already set).

## Validation

Datasets are validated before any model is loaded: empty files, malformed JSON,
missing fields and non-string fields are rejected with a clear error.

## Configs (`configs/*.json`)

A config is a single JSON object. Unknown keys are rejected; keys starting with
`_` are ignored and can hold comments. Main keys (see `r1_grpo.training` —
`GRPO_DEFAULTS`, `SFT_DEFAULTS`, `GENERATION_DEFAULTS` — for the authoritative
list and defaults):

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `"tiny"` | `"tiny"` offline CPU fixture or `"hf"` (`transformers` AutoModelForCausalLM; `uv sync --extra hf`) |
| `stage` | — | `zero` / `reasoning` / `all` for `grpo`; `cold_start` / `mixed_sft` / `distill` for `sft` |
| `model_name_or_path` | — | HF model ID or a saved `output_dir`. For `tiny`, omit it to build a fresh fixture initialised from `seed`. |
| `dataset` | — | JSONL path |
| `output_dir` | — | checkpoint directory (existing checkpoints are not overwritten) |
| `seed`, `device` | `42`, `"cpu"` | |
| `learning_rate`, `batch_size` | `1e-4`, `1` | |
| `max_steps` / `epochs` | `2` / `2` | RL steps / SFT epochs |
| `group_size`, `update_epochs` | `4`, `1` | At least 2 completions per GRPO group; inner updates per rollout |
| `clip_epsilon`, `beta`, `reduction` | `0.2`, `0.04`, `"sequence"` | GRPO objective (`"token"` = practical variant) |
| `max_prompt_tokens` | `512` | prompt budget including the template; overlong prompts fail explicitly |
| `max_new_tokens` | RL `64`; sampling `32768` | completion budget |
| `max_seq_length` | RL/SFT `1024`; sampling `33280` | total length limit |
| `temperature`, `top_p` | RL `1.0`, `1.0`; sampling `0.6`, `0.95` | |
| `num_samples` | `16` (allowed 4-64) | responses per prompt for `sample` |
| `max_grad_norm` | `1.0` | gradient clipping (RL and SFT) |
| `callbacks` | `{}` | `{"verifier", "language_scorer", "helpfulness", "harmlessness"}` → `"module:function"` |

### Provided configs

| File | Command | Stage |
| --- | --- | --- |
| `configs/r1_zero.json` | `grpo` | R1-Zero, fresh tiny base |
| `configs/r1_stage1_cold_start_sft.json` | `sft` | R1 stage 1, fresh tiny base |
| `configs/r1_stage2_reasoning_rl.json` | `grpo` | R1 stage 2, from stage 1 |
| `configs/r1_stage3_sample.json` | `sample` | R1 stage 3 candidates, from stage 2 |
| `configs/r1_stage3_mixed_sft.json` | `sft` | R1 stage 3 SFT, **fresh base**, 2 epochs |
| `configs/r1_stage4_all_rl.json` | `grpo` | R1 stage 4, from stage 3, toy callbacks |
| `configs/distill_sft.json` | `sft` | SFT-only distillation into a fresh tiny student |
| `configs/eval_toy.json` | `sample` | toy evaluation sampling (0.6 / 0.95, 16 samples) |
| `configs/hf/r1_zero_qwen2.5_math_1.5b.json` | `grpo` | R1-Zero on the small HF substitute base |
| `configs/hf/distill_sft_qwen2.5_math_1.5b.json` | `sft` | distillation into Qwen2.5-Math-1.5B |
| `configs/hf/eval_paper_v1.json` | `sample` | paper-v1 sampling: 0.6 / 0.95 / 32,768 tokens, 16 samples |

The tiny configs raise `max_prompt_tokens` / `max_seq_length` to 1024 / 2048
to leave room for longer questions with its byte tokenizer.

`configs/hf/eval_paper_v1.json` points at `runs/hf-r1-stage4-all-rl` as a
placeholder; set `model_name_or_path` to the checkpoint you want to evaluate.
The model must support a context of at least `max_seq_length`.

## Callbacks

A callback spec is `"package.module:function"`, imported with the current
working directory on `sys.path`. Signatures:

```python
verifier(example: dict, response: str) -> float
language_scorer(reasoning_text: str) -> float
helpfulness(prompt: str, final_text: str) -> float
harmlessness(prompt: str, full_response: str) -> float
```

Return values must be finite. `examples/callbacks.py` contains **untrained
illustrations** (`toy_helpfulness`, `toy_harmlessness`, `ascii_letter_ratio`,
`multiple_choice_verifier`); they are not reward models. A code-correctness
verifier should submit code to an external sandbox and return its verdict; never
execute generated code inside the training process.

Use the same correctness verifier throughout the workflow: set
`callbacks.verifier` in the GRPO config, then pass `--verifier module:function`
to both `reject` and `evaluate`. Those two commands treat scores of at least 1
as correct; training uses the numeric score itself. For code tasks, pass@1
comes from the judge; textual consensus remains a vote over extracted strings,
not a claim of semantic program equivalence.

Checkpoints contain model/tokenizer files, `r1_grpo_metadata.json` (resolved
configuration, initialization lineage, data hash and library versions), and
`metrics.json`. They support inference and initialization of subsequent stages;
optimizer state is not saved for exact interrupted-run resumption.
