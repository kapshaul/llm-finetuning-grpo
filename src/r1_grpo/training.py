"""Pure-PyTorch SFT, GRPO, and sampling loops for the DeepSeek-R1 pipeline.

Stages (DeepSeek-R1 v1, arXiv:2501.12948):

* ``train_grpo`` stage ``zero``: R1-Zero pure RL from a base model with rule
  accuracy + format rewards.
* ``train_sft`` stage ``cold_start``: SFT on a small cold-start set.
* ``train_grpo`` stage ``reasoning``: reasoning RL with accuracy + language
  consistency rewards.
* ``generate_candidates`` + rejection sampling (``rejection.py``), then
  ``train_sft`` stage ``mixed_sft`` from the *original base* model (paper: 2
  epochs) on reasoning + general data.
* ``train_grpo`` stage ``all``: RL for all scenarios.
* ``train_sft`` stage ``distill``: SFT-only distillation of a smaller model.

The paper does not disclose optimizer settings, reward weights, group size, the
language classifier, reward models, or training data. Every default here
(AdamW without weight decay, learning rate, group size, clipping, beta,
max_grad_norm, sequence lengths) is an explicit substitute, not a reproduction
of the paper's values. Weight decay is fixed at 0 so that a GRPO step whose
groups all have tied rewards (zero advantage) and whose policy still equals the
reference leaves the parameters unchanged.

Stage ``zero`` refuses local r1_grpo checkpoints whose lineage contains any
stage other than ``zero`` (e.g. SFT or later RL), so SFT+RL is never labelled
R1-Zero. This guard only sees r1_grpo metadata: a raw hub ID or a local
directory without that metadata cannot be proven to be a base model and is
accepted as-is.

GRPO rollouts are sampled with temperature 1 and top_p 1 so the behaviour
distribution equals the policy distribution used for old/current
log-probabilities. Old log-probs are computed once per rollout batch and kept
frozen across update epochs; the KL reference is a frozen copy of the policy
taken at the start of the stage.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from . import models
from .core import group_advantages, grpo_loss
from .data import read_jsonl, validate_examples, write_jsonl
from .prompts import format_prompt
from .rewards import score_response

GRPO_STAGES = ("zero", "reasoning", "all")
SFT_STAGES = ("cold_start", "mixed_sft", "distill")
CALLBACK_NAMES = ("verifier", "language_scorer", "helpfulness", "harmlessness")
# Hooks consumed by each RL stage; configuring any other hook is an error rather
# than being silently ignored. Zero uses rule rewards only (an external
# rule-based verifier, e.g. a sandboxed code judge, may replace accuracy).
STAGE_CALLBACKS = {
    "zero": ("verifier",),
    "reasoning": ("verifier", "language_scorer"),
    "all": ("verifier", "helpfulness", "harmlessness"),
}
METADATA_FILE = "r1_grpo_metadata.json"
METRICS_FILE = "metrics.json"

_REQUIRED = object()

_COMMON_DEFAULTS: dict[str, Any] = {
    "backend": "tiny",
    "model_name_or_path": None,
    "dataset": _REQUIRED,
    "seed": 42,
    "device": "cpu",
    "max_prompt_tokens": 512,
    "max_seq_length": 1024,
}
GRPO_DEFAULTS: dict[str, Any] = {
    **_COMMON_DEFAULTS,
    "stage": _REQUIRED,
    "output_dir": _REQUIRED,
    "learning_rate": 1e-4,
    "batch_size": 1,
    "max_steps": 2,
    "group_size": 4,
    "update_epochs": 1,
    "clip_epsilon": 0.2,
    "beta": 0.04,
    "reduction": "sequence",
    "max_new_tokens": 64,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_grad_norm": 1.0,
    "callbacks": {},
}
SFT_DEFAULTS: dict[str, Any] = {
    **_COMMON_DEFAULTS,
    "stage": _REQUIRED,
    "output_dir": _REQUIRED,
    "learning_rate": 1e-4,
    "batch_size": 1,
    "epochs": 2,
    "max_steps": None,
    "max_grad_norm": 1.0,
}
GENERATION_DEFAULTS: dict[str, Any] = {
    **_COMMON_DEFAULTS,
    "max_new_tokens": 32768,
    "max_seq_length": 32768 + 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "num_samples": 16,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _resolve_config(config: dict, defaults: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge ``config`` over ``defaults`` rejecting unknown or missing keys.

    Keys starting with ``_`` are treated as comments and ignored.
    """
    if not isinstance(config, dict):
        raise TypeError(f"{name} config must be a dict, got {type(config).__name__}")
    keys = {k for k in config if not (isinstance(k, str) and k.startswith("_"))}
    unknown = sorted(str(k) for k in keys - defaults.keys())
    if unknown:
        raise ValueError(f"unknown {name} config key(s): {', '.join(unknown)}; allowed: {', '.join(sorted(defaults))}")
    cfg = {key: config.get(key, default) for key, default in defaults.items()}
    missing = sorted(k for k, v in cfg.items() if v is _REQUIRED)
    if missing:
        raise ValueError(f"missing required {name} config key(s): {', '.join(missing)}")

    if cfg["backend"] not in models.BACKENDS:
        raise ValueError(f"backend must be one of {models.BACKENDS}, got {cfg['backend']!r}")
    for key in ("dataset",) + (("output_dir",) if "output_dir" in cfg else ()):
        _check_str(cfg, key)
    if cfg["model_name_or_path"] is not None:
        _check_str(cfg, "model_name_or_path")
    elif cfg["backend"] == "hf":
        raise ValueError("backend 'hf' requires model_name_or_path")
    _check_str(cfg, "device")
    try:
        torch.device(cfg["device"])
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {cfg['device']!r}") from exc

    _check_int(cfg, "seed", minimum=0)
    _check_int(cfg, "max_prompt_tokens", minimum=1)
    _check_int(cfg, "max_seq_length", minimum=2)
    for key in ("batch_size", "update_epochs", "epochs", "max_new_tokens"):
        if key in cfg:
            _check_int(cfg, key, minimum=1)
    if "group_size" in cfg:
        # A single-sample group always has zero group-relative advantage.
        _check_int(cfg, "group_size", minimum=2)
    if "max_steps" in cfg and not (cfg["max_steps"] is None and "epochs" in cfg):
        _check_int(cfg, "max_steps", minimum=1)
    if "num_samples" in cfg:
        # Paper evaluation uses k samples per question; supported range 4-64.
        _check_int(cfg, "num_samples", minimum=4, maximum=64)
    for key in ("learning_rate", "max_grad_norm", "clip_epsilon", "temperature"):
        if key in cfg:
            _check_float(cfg, key, minimum=0.0, inclusive=False)
    if "clip_epsilon" in cfg and cfg["clip_epsilon"] >= 1.0:
        raise ValueError(f"config key 'clip_epsilon' must be < 1, got {cfg['clip_epsilon']}")
    if "beta" in cfg:
        _check_float(cfg, "beta", minimum=0.0, inclusive=True)
    if "top_p" in cfg:
        _check_float(cfg, "top_p", minimum=0.0, inclusive=False)
        if cfg["top_p"] > 1.0:
            raise ValueError("top_p must be in (0, 1]")
    if "reduction" in cfg and cfg["reduction"] not in ("sequence", "token"):
        raise ValueError("reduction must be 'sequence' (paper default) or 'token'")
    return cfg


def _check_str(cfg: dict, key: str) -> None:
    if not isinstance(cfg[key], str) or not cfg[key].strip():
        raise ValueError(f"config key {key!r} must be a non-empty string")


def _check_int(cfg: dict, key: str, *, minimum: int, maximum: int | None = None) -> None:
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config key {key!r} must be an integer, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"config key {key!r} must be {bound}, got {value}")


def _check_float(cfg: dict, key: str, *, minimum: float, inclusive: bool) -> None:
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"config key {key!r} must be a finite number, got {value!r}")
    if value < minimum or (not inclusive and value == minimum):
        op = ">=" if inclusive else ">"
        raise ValueError(f"config key {key!r} must be {op} {minimum}, got {value}")
    cfg[key] = float(value)


def _load_callbacks(spec: Any, allowed: tuple[str, ...], stage: str) -> dict[str, Callable]:
    """Resolve ``{"name": "module:function"}`` (or callables) into callables."""
    if not isinstance(spec, dict):
        raise ValueError("callbacks must be a mapping of hook name to 'module:function'")
    resolved: dict[str, Callable] = {}
    for name, target in spec.items():
        if name not in CALLBACK_NAMES:
            raise ValueError(f"unknown callback {name!r}; expected one of {CALLBACK_NAMES}")
        if name not in allowed:
            raise ValueError(f"callback {name!r} is not used by stage {stage!r}")
        if callable(target):
            resolved[name] = target
            continue
        if not isinstance(target, str) or target.count(":") != 1:
            raise ValueError(f"callback {name!r} must be 'module:function', got {target!r}")
        module_name, attr = target.split(":")
        try:
            func = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot import callback {name!r} from {target!r}: {exc}") from exc
        if not callable(func):
            raise ValueError(f"callback {name!r} target {target!r} is not callable")
        resolved[name] = func
    return resolved


def _jsonable_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    if "callbacks" in out:
        out["callbacks"] = {
            name: target if isinstance(target, str) else f"<callable {getattr(target, '__qualname__', repr(target))}>"
            for name, target in out["callbacks"].items()
        }
    return out


# ---------------------------------------------------------------------------
# Data, checkpoints, provenance
# ---------------------------------------------------------------------------
def _load_dataset(path: str, *, supervised: bool, require_answer: bool = True) -> list[dict]:
    records = read_jsonl(path)
    validate_examples(records, supervised=supervised, require_answer=require_answer)
    if not records:
        raise ValueError(f"dataset {path} is empty")
    return records


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_fresh_output_dir(path: str | Path) -> None:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing non-empty output path {path}")


def _source_metadata(model_name_or_path: str | None) -> dict | None:
    if model_name_or_path is None:
        return None
    meta_file = Path(model_name_or_path) / METADATA_FILE
    if not meta_file.is_file():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _check_zero_source(source_meta: dict | None, model_name_or_path: str | None) -> None:
    """Reject known r1_grpo checkpoints that went through any non-``zero`` stage.

    Only r1_grpo metadata is visible here; sources without it are accepted
    because they cannot be proven either way.
    """
    if source_meta is None:
        return
    stages = [entry.get("stage") for entry in source_meta.get("lineage", [])]
    stages.append(source_meta.get("stage"))
    foreign = [stage for stage in stages if stage not in ("base", "zero")]
    if foreign:
        raise ValueError(
            "stage 'zero' (R1-Zero) must start from a base model (paper Section 2.2), "
            f"but {model_name_or_path} is an r1_grpo checkpoint with lineage stage(s) "
            f"{sorted({str(stage) for stage in foreign})}"
        )


def _lineage(source_meta: dict | None, source: str) -> list[dict]:
    if source_meta is None:
        return [{"stage": "base", "source": source}]
    return list(source_meta.get("lineage", [])) + [{"stage": source_meta.get("stage"), "source": source}]


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def _write_checkpoint(policy: models.Policy, output_dir: str, metadata: dict, metrics: dict) -> None:
    _ensure_fresh_output_dir(output_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    # allow_nan=False makes any non-finite metric a hard error instead of a silent NaN.
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / METRICS_FILE).write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _example_label(record: dict, index: int) -> str:
    return f"example {record['id']!r}" if "id" in record else f"example #{index}"


def _encode_prompt_checked(policy: models.Policy, record: dict, index: int, limit: int) -> list[int]:
    ids = policy.encode_prompt(format_prompt(record["prompt"], domain=record.get("domain", "reasoning")))
    if len(ids) > limit:
        raise ValueError(
            f"{_example_label(record, index)}: prompt has {len(ids)} tokens, exceeding max_prompt_tokens={limit}"
        )
    return ids


def _snapshot(policy: models.Policy):
    """Frozen copy of the initial tiny model for parameter-delta reporting."""
    return policy.frozen_copy() if policy.backend == "tiny" else None


def _delta(initial, policy: models.Policy) -> float | None:
    if initial is None:
        return None
    return models.parameter_l2_distance(initial.model, policy.model)


def _optimizer_step(
    policy: models.Policy, optimizer: torch.optim.Optimizer, loss: torch.Tensor, max_norm: float
) -> float:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in policy.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(grad_norm)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def train_sft(config: dict) -> dict:
    """Supervised fine-tuning on ``{"prompt", "response"}`` rows; returns a summary."""
    summary, _ = _train_sft(config)
    return summary


def _train_sft(config: dict) -> tuple[dict, models.Policy]:
    cfg = _resolve_config(config, SFT_DEFAULTS, "train_sft")
    stage = cfg["stage"]
    if stage not in SFT_STAGES:
        raise ValueError(f"train_sft stage must be one of {SFT_STAGES}, got {stage!r}")
    _ensure_fresh_output_dir(cfg["output_dir"])
    records = _load_dataset(cfg["dataset"], supervised=True)
    source_meta = _source_metadata(cfg["model_name_or_path"])
    if stage == "mixed_sft" and source_meta is not None:
        raise ValueError(
            "mixed_sft must start from the original base model (paper Section 2.3.3), "
            f"but {cfg['model_name_or_path']} is an r1_grpo {source_meta.get('stage')!r} checkpoint"
        )

    _seed_everything(cfg["seed"])
    policy = models.load_policy(cfg["backend"], cfg["model_name_or_path"], cfg["device"], cfg["seed"])
    initial = _snapshot(policy)

    encoded: list[tuple[list[int], list[int]]] = []
    truncated = 0
    for index, record in enumerate(records):
        prompt_ids = _encode_prompt_checked(policy, record, index, cfg["max_prompt_tokens"])
        completion = policy.encode_text(record["response"]) + [policy.eos_token_id]
        room = cfg["max_seq_length"] - len(prompt_ids)
        if room < 1:
            raise ValueError(
                f"{_example_label(record, index)}: truncation to max_seq_length="
                f"{cfg['max_seq_length']} would leave no completion labels"
            )
        if len(completion) > room:
            truncated += 1
            completion = completion[:room]
        encoded.append((prompt_ids, completion))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"], weight_decay=0.0)
    rng = random.Random(cfg["seed"])
    steps_per_epoch = math.ceil(len(encoded) / cfg["batch_size"])
    history: list[dict[str, float]] = []
    policy.model.train()
    started = time.perf_counter()
    step = 0
    for epoch in range(cfg["epochs"]):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for begin in range(0, len(order), cfg["batch_size"]):
            if cfg["max_steps"] is not None and step >= cfg["max_steps"]:
                break
            batch = [encoded[i] for i in order[begin : begin + cfg["batch_size"]]]
            log_probs, mask = models.completion_log_probs(policy, [b[0] for b in batch], [b[1] for b in batch])
            num_tokens = mask.sum()
            loss = -(log_probs * mask).sum() / num_tokens
            grad_norm = _optimizer_step(policy, optimizer, loss, cfg["max_grad_norm"])
            step += 1
            history.append(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "grad_norm": grad_norm,
                    "label_tokens": int(num_tokens.item()),
                }
            )
    policy.model.eval()
    if not history:
        raise ValueError("SFT performed no optimizer steps")

    summary = {
        "kind": "sft",
        "stage": stage,
        "backend": policy.backend,
        "output_dir": cfg["output_dir"],
        "steps": step,
        # ``epochs`` is the configured count; max_steps may stop training earlier.
        "epochs": cfg["epochs"],
        "completed_epochs": step // steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "num_examples": len(encoded),
        "truncated_examples": truncated,
        "first_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "parameter_delta_l2": _delta(initial, policy),
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata = {
        "package": "r1_grpo",
        "kind": "sft",
        "stage": stage,
        "backend": policy.backend,
        "source_model": policy.source,
        "lineage": _lineage(source_meta, policy.source),
        "dataset": cfg["dataset"],
        "dataset_sha256": _file_sha256(cfg["dataset"]),
        "config": _jsonable_config(cfg),
        "environment": _environment(),
        "created_at": datetime.now(UTC).isoformat(),
        "notes": [
            "Optimizer (AdamW, weight_decay=0) and all hyperparameters are documented "
            "substitutes; the paper does not disclose them.",
            "Loss is mean next-token cross-entropy over response tokens plus EOS; "
            "prompt and padding tokens are masked.",
        ],
    }
    _write_checkpoint(policy, cfg["output_dir"], metadata, {"summary": summary, "history": history})
    return summary, policy


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
def train_grpo(config: dict) -> dict:
    """On-policy GRPO (paper Equations 1-3) with rule/hook rewards; returns a summary."""
    summary, _ = _train_grpo(config)
    return summary


def _train_grpo(config: dict) -> tuple[dict, models.Policy]:
    cfg = _resolve_config(config, GRPO_DEFAULTS, "train_grpo")
    stage = cfg["stage"]
    if stage not in GRPO_STAGES:
        raise ValueError(f"train_grpo stage must be one of {GRPO_STAGES}, got {stage!r}")
    if cfg["temperature"] != 1.0 or cfg["top_p"] != 1.0:
        raise ValueError(
            "GRPO rollouts must use temperature=1.0 and top_p=1.0 so the sampling "
            "distribution matches the old/current policy log-probabilities "
            "(no behaviour-policy correction is implemented)"
        )
    _ensure_fresh_output_dir(cfg["output_dir"])
    callbacks = _load_callbacks(cfg["callbacks"], STAGE_CALLBACKS[stage], stage)
    # An external verifier scores reasoning examples itself (e.g. code tests kept
    # as example metadata), so a reference answer is only required without one.
    records = _load_dataset(cfg["dataset"], supervised=False, require_answer="verifier" not in callbacks)
    has_general = any(r.get("domain", "reasoning") == "general" for r in records)
    if has_general and stage != "all":
        raise ValueError(f"stage {stage!r} accepts reasoning examples only; found domain='general'")
    if has_general and not {"helpfulness", "harmlessness"} <= callbacks.keys():
        raise ValueError(
            "general-domain examples require 'helpfulness' and 'harmlessness' callbacks "
            "(preference models are not bundled)"
        )
    source_meta = _source_metadata(cfg["model_name_or_path"])
    if stage == "zero":
        _check_zero_source(source_meta, cfg["model_name_or_path"])

    _seed_everything(cfg["seed"])
    policy = models.load_policy(cfg["backend"], cfg["model_name_or_path"], cfg["device"], cfg["seed"])
    # RL runs in eval mode (dropout off) while gradients stay enabled.
    policy.model.eval()
    initial = _snapshot(policy)
    reference = policy.frozen_copy() if cfg["beta"] > 0 else None

    prompt_ids = [
        _encode_prompt_checked(policy, record, index, cfg["max_prompt_tokens"]) for index, record in enumerate(records)
    ]
    budgets = []
    for index, ids in enumerate(prompt_ids):
        budget = min(cfg["max_new_tokens"], cfg["max_seq_length"] - len(ids))
        if budget < 1:
            raise ValueError(
                f"{_example_label(records[index], index)}: no room for completion tokens "
                f"within max_seq_length={cfg['max_seq_length']}"
            )
        budgets.append(budget)

    # No weight decay: zero advantages with policy == reference must not move parameters.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"], weight_decay=0.0)
    rng = random.Random(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    order: list[int] = []
    B, G = cfg["batch_size"], cfg["group_size"]
    history: list[dict[str, Any]] = []
    policy_has_updated = False
    started = time.perf_counter()

    for step in range(1, cfg["max_steps"] + 1):
        batch_indices = []
        for _ in range(B):
            if not order:
                order = list(range(len(records)))
                rng.shuffle(order)
            batch_indices.append(order.pop())

        # --- rollouts (exact sampled token IDs are kept for training) ---
        flat_prompts: list[list[int]] = []
        flat_completions: list[list[int]] = []
        rewards: list[list[float]] = []
        components: dict[str, list[float]] = {}
        stopped = 0
        for index in batch_indices:
            samples = models.sample_completions(policy, prompt_ids[index], G, budgets[index], 1.0, 1.0, generator)
            group_rewards = []
            for sample in samples:
                scores = score_response(records[index], sample.text, stage=stage, **callbacks)
                total = float(scores["total"])
                if not math.isfinite(total):
                    raise FloatingPointError(f"non-finite reward {total} for {records[index]}")
                group_rewards.append(total)
                for key, value in scores.items():
                    components.setdefault(key, []).append(float(value))
                flat_prompts.append(prompt_ids[index])
                flat_completions.append(sample.token_ids)
                stopped += int(sample.stopped_on_eos)
            rewards.append(group_rewards)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=policy.device)
        advantages = group_advantages(reward_tensor)

        with torch.no_grad():
            old_lp, mask = models.completion_log_probs(policy, flat_prompts, flat_completions)
            ref_lp = None
            if reference is not None:
                # Until the first nonzero update, actor and reference weights are
                # identical. Reuse their identical scores rather than introducing
                # rounding differences from frozen-parameter inference kernels.
                ref_lp = (
                    models.completion_log_probs(reference, flat_prompts, flat_completions)[0]
                    if policy_has_updated
                    else old_lp.clone()
                )
        T = mask.shape[-1]
        old_lp = old_lp.view(B, G, T)
        mask = mask.view(B, G, T)
        if ref_lp is not None:
            ref_lp = ref_lp.view(B, G, T)

        # --- policy updates against frozen old/ref log-probs ---
        epoch_stats = []
        for _ in range(cfg["update_epochs"]):
            log_probs, _ = models.completion_log_probs(policy, flat_prompts, flat_completions)
            log_probs = log_probs.view(B, G, T)
            ref = ref_lp if ref_lp is not None else log_probs.detach()
            loss, loss_metrics = grpo_loss(
                log_probs,
                old_lp,
                ref,
                mask,
                advantages,
                clip_epsilon=cfg["clip_epsilon"],
                beta=cfg["beta"],
                reduction=cfg["reduction"],
            )
            grad_norm = _optimizer_step(policy, optimizer, loss, cfg["max_grad_norm"])
            policy_has_updated = policy_has_updated or grad_norm > 0.0
            epoch_stats.append({"loss": float(loss.detach()), "grad_norm": grad_norm, **loss_metrics})

        lengths = mask.sum(-1)
        entry: dict[str, Any] = {
            "step": step,
            "reward_mean": float(reward_tensor.mean()),
            "reward_std": float(reward_tensor.std(correction=0)),
            "reward_components": {k: sum(v) / len(v) for k, v in components.items()},
            "groups_with_nonzero_advantage": int((advantages.abs() > 0).any(-1).sum()),
            "completion_tokens_mean": float(lengths.float().mean()),
            "truncation_rate": 1.0 - stopped / (B * G),
            "updates": epoch_stats,
        }
        history.append(entry)

    total_samples = B * G * len(history)
    summary = {
        "kind": "grpo",
        "stage": stage,
        "backend": policy.backend,
        "output_dir": cfg["output_dir"],
        "steps": len(history),
        "group_size": G,
        "reduction": cfg["reduction"],
        "mean_reward": sum(h["reward_mean"] for h in history) / len(history),
        "final_reward_mean": history[-1]["reward_mean"],
        "final_loss": history[-1]["updates"][-1]["loss"],
        "truncation_rate": sum(h["truncation_rate"] * B * G for h in history) / total_samples,
        "groups_with_nonzero_advantage": sum(h["groups_with_nonzero_advantage"] for h in history),
        "parameter_delta_l2": _delta(initial, policy),
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata = {
        "package": "r1_grpo",
        "kind": "grpo",
        "stage": stage,
        "backend": policy.backend,
        "source_model": policy.source,
        "lineage": _lineage(source_meta, policy.source),
        "dataset": cfg["dataset"],
        "dataset_sha256": _file_sha256(cfg["dataset"]),
        "config": _jsonable_config(cfg),
        "environment": _environment(),
        "created_at": datetime.now(UTC).isoformat(),
        "notes": [
            "Objective reduction 'sequence' sums completion-token log-probs before the "
            "ratio and KL (paper Eq. 1-3); 'token' is a practical variant from the "
            "cited GRPO work.",
            "Group advantages use population std (correction=0); tied groups get zero advantage.",
            "Rollouts use temperature=1, top_p=1 (full support, on-policy).",
            "Optimizer (AdamW, weight_decay=0), group size, beta, clip epsilon, and reward "
            "weights are documented substitutes; the paper does not disclose them.",
            "The stage 'zero' base-model guard only inspects r1_grpo checkpoint metadata; "
            "sources without it cannot be verified as base models.",
        ],
    }
    _write_checkpoint(policy, cfg["output_dir"], metadata, {"summary": summary, "history": history})
    return summary, policy


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def generate_candidates(config: dict) -> list[dict]:
    """Sample ``num_samples`` responses per prompt for rejection sampling/evaluation.

    Each output record copies the input example and adds ``responses``.
    Defaults follow the paper's evaluation settings (temperature 0.6, top_p
    0.95, up to 32768 new tokens, 16 samples).
    """
    cfg = _resolve_config(config, GENERATION_DEFAULTS, "generate_candidates")
    # Sampling needs no ground truth; answers, when present, are still type-checked.
    records = _load_dataset(cfg["dataset"], supervised=False, require_answer=False)
    _seed_everything(cfg["seed"])
    policy = models.load_policy(cfg["backend"], cfg["model_name_or_path"], cfg["device"], cfg["seed"])
    policy.model.eval()
    generator = torch.Generator().manual_seed(cfg["seed"])
    outputs = []
    for index, record in enumerate(records):
        ids = _encode_prompt_checked(policy, record, index, cfg["max_prompt_tokens"])
        budget = min(cfg["max_new_tokens"], cfg["max_seq_length"] - len(ids))
        if budget < 1:
            raise ValueError(
                f"{_example_label(record, index)}: no room for completion tokens "
                f"within max_seq_length={cfg['max_seq_length']}"
            )
        samples = models.sample_completions(
            policy, ids, cfg["num_samples"], budget, cfg["temperature"], cfg["top_p"], generator
        )
        row = dict(record)
        row["responses"] = [s.text for s in samples]
        outputs.append(row)
    return outputs


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
_SMOKE_PROBLEMS = [
    ("What is 2 + 3?", "2 + 3", "5"),
    ("What is 7 - 4?", "7 - 4", "3"),
    ("What is 3 * 4?", "3 * 4", "12"),
    ("What is 9 / 3?", "9 / 3", "3"),
]


def _assert_finite(value: Any, path: str = "summary") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            _assert_finite(item, f"{path}[{i}]")


def smoke(output_dir: str) -> dict:
    """Offline end-to-end check with the tiny GRU fixture.

    Runs tiny cold-start SFT, real on-policy R1-Zero GRPO, sampling,
    evaluation, and checkpoint reloads. A random tiny model is expected to earn
    (near) zero reward; this checks plumbing, not learned reasoning.
    """
    root = Path(output_dir)
    _ensure_fresh_output_dir(root)
    root.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    try:
        from .evaluation import evaluate_records

        data_dir = root / "data"
        sft_path, rl_path = data_dir / "sft.jsonl", data_dir / "rl.jsonl"
        write_jsonl(
            sft_path,
            [
                {
                    "id": f"smoke-{i}",
                    "prompt": question,
                    "response": f"<think>\n{expr} = {answer}\n</think>\n<answer>{answer}</answer>",
                }
                for i, (question, expr, answer) in enumerate(_SMOKE_PROBLEMS)
            ],
        )
        write_jsonl(
            rl_path,
            [
                {"id": f"smoke-{i}", "prompt": question, "answer": answer, "domain": "reasoning"}
                for i, (question, _, answer) in enumerate(_SMOKE_PROBLEMS)
            ],
        )
        lengths = {"max_prompt_tokens": 2048, "max_seq_length": 4096}
        sft_dir, grpo_dir = str(root / "cold_start"), str(root / "zero")
        sft_summary = train_sft(
            {
                "stage": "cold_start",
                "dataset": str(sft_path),
                "output_dir": sft_dir,
                "learning_rate": 3e-3,
                "batch_size": 2,
                "epochs": 2,
                **lengths,
            }
        )
        grpo_summary = train_grpo(
            {
                "stage": "zero",
                "dataset": str(rl_path),
                "output_dir": grpo_dir,
                "batch_size": 2,
                "group_size": 4,
                "max_steps": 2,
                "max_new_tokens": 32,
                **lengths,
            }
        )
        candidates = generate_candidates(
            {"model_name_or_path": grpo_dir, "dataset": str(rl_path), "num_samples": 4, "max_new_tokens": 32, **lengths}
        )
        candidates_path = root / "candidates.jsonl"
        write_jsonl(candidates_path, candidates)
        evaluation = evaluate_records(candidates)
        (root / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

        fresh = models.load_policy("tiny", None, "cpu", 42)
        sft_reloaded = models.load_policy("tiny", sft_dir, "cpu", 42)
        grpo_reloaded = models.load_policy("tiny", grpo_dir, "cpu", 42)
        probe_prompt = sft_reloaded.encode_prompt(format_prompt(_SMOKE_PROBLEMS[0][0]))
        probe_completion = sft_reloaded.encode_text("<answer>5</answer>") + [sft_reloaded.eos_token_id]
        with torch.no_grad():
            probe_lp, _ = models.completion_log_probs(sft_reloaded, [probe_prompt], [probe_completion])
        summary = {
            "backend": "tiny (byte-level GRU test fixture, not DeepSeek-V3)",
            "sft": sft_summary,
            "grpo": grpo_summary,
            "generation": {"path": str(candidates_path), "records": len(candidates)},
            "evaluation": evaluation,
            "reload": {
                "sft_delta_l2_vs_fresh_init": models.parameter_l2_distance(fresh.model, sft_reloaded.model),
                "grpo_delta_l2_vs_fresh_init": models.parameter_l2_distance(fresh.model, grpo_reloaded.model),
                "sft_probe_log_prob": float(probe_lp.sum()),
            },
            "note": (
                "Offline plumbing check with a randomly initialized tiny model; rewards "
                "near zero are expected and no learned reasoning is claimed."
            ),
        }
        _assert_finite(summary)
        (root / "smoke_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        torch.set_num_threads(previous_threads)
