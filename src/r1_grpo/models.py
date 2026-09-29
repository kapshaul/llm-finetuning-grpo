"""Policy model backends, exact-token log-probabilities, and manual sampling.

Two backends are supported:

* ``tiny``: a byte-level GRU language model. It is a fully offline CPU **test
  fixture** used for smoke tests and unit tests. It is *not* the DeepSeek-V3
  architecture and is not expected to learn reasoning.
* ``hf``: any ``transformers.AutoModelForCausalLM`` checkpoint (full-weight
  fine-tuning on a single device). ``transformers`` is imported lazily and
  ``trust_remote_code`` is always ``False``.

Sampling is implemented manually from raw full-vocabulary logits (no inherited
``generate()`` processors such as top-k, repetition penalties, or forced/
suppressed tokens). Sampled token IDs are returned verbatim so training never
decodes and re-tokenizes actions.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

BACKENDS = ("tiny", "hf")
TINY_CONFIG_FILE = "tiny_config.json"
TINY_WEIGHTS_FILE = "tiny_model.pt"
TINY_TOKENIZER_FILE = "tiny_tokenizer.json"
TINY_ARCHITECTURE = "tiny-byte-gru-test-fixture"


class ByteTokenizer:
    """UTF-8 byte tokenizer with PAD/BOS/EOS special tokens (IDs 256, 257, 258)."""

    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    vocab_size = 259
    _special_names = {256: "<pad>", 257: "<bos>", 258: "<eos>"}

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.bos_token_id, *ids] if add_bos else ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        parts: list[str] = []
        buffer = bytearray()
        for token in ids:
            token = int(token)
            if 0 <= token < 256:
                buffer.append(token)
                continue
            if token not in self._special_names:
                raise ValueError(f"token id {token} is outside the byte tokenizer vocabulary")
            if not skip_special_tokens:
                parts.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()
                parts.append(self._special_names[token])
        parts.append(buffer.decode("utf-8", errors="replace"))
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "utf8-bytes",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }


class TinyGRULM(nn.Module):
    """Small byte-level GRU language model (offline test fixture, not V3)."""

    def __init__(
        self,
        vocab_size: int = ByteTokenizer.vocab_size,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.config = {
            "architecture": TINY_ARCHITECTURE,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output, state = self.gru(self.embed(input_ids), state)
        return self.head(output), state


@dataclass
class SampledCompletion:
    """One sampled completion. ``token_ids`` are the exact sampled actions."""

    token_ids: list[int]
    text: str
    stopped_on_eos: bool


class Policy:
    """Backend-agnostic wrapper around a causal LM and its tokenizer."""

    def __init__(
        self,
        backend: str,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        source: str,
    ) -> None:
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.source = source
        if backend == "tiny":
            self.eos_token_id = ByteTokenizer.eos_token_id
            self.pad_token_id = ByteTokenizer.pad_token_id
        else:
            if tokenizer.eos_token_id is None:
                raise ValueError("the Hugging Face tokenizer defines no eos_token_id")
            self.eos_token_id = int(tokenizer.eos_token_id)
            # Padding positions are always masked explicitly, so EOS is a safe pad.
            pad = tokenizer.pad_token_id
            self.pad_token_id = int(pad) if pad is not None else self.eos_token_id

    # -- tokenization -----------------------------------------------------
    def encode_prompt(self, text: str) -> list[int]:
        """Encode a prompt, including BOS (tiny) or the tokenizer's special tokens (hf)."""
        if self.backend == "tiny":
            ids = self.tokenizer.encode(text, add_bos=True)
        else:
            ids = list(self.tokenizer(text, add_special_tokens=True)["input_ids"])
        if not ids:
            raise ValueError("prompt encodes to zero tokens; a causal LM needs at least one")
        return ids

    def encode_text(self, text: str) -> list[int]:
        """Encode continuation text without special tokens."""
        if self.backend == "tiny":
            return self.tokenizer.encode(text)
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    # -- forward passes ---------------------------------------------------
    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Full-sequence logits for right-padded inputs ``[N, L] -> [N, L, V]``.

        Right padding keeps real-token positions at ``0..len-1`` for both the
        causal GRU and HF default position ids, so padding never influences
        real-token logits.
        """
        if self.backend == "tiny":
            logits, _ = self.model(input_ids)
            return logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    def start(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Run the (unpadded) prompt batch and return next-token logits and state."""
        if self.backend == "tiny":
            logits, state = self.model(input_ids)
            return logits[:, -1], state
        mask = torch.ones_like(input_ids)
        out = self.model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        return out.logits[:, -1], (out.past_key_values, input_ids.shape[1])

    def advance(self, tokens: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        """Feed one token per row and return the following next-token logits."""
        if self.backend == "tiny":
            logits, state = self.model(tokens[:, None], state)
            return logits[:, -1], state
        past, length = state
        mask = torch.ones((tokens.shape[0], length + 1), dtype=torch.long, device=tokens.device)
        out = self.model(input_ids=tokens[:, None], attention_mask=mask, past_key_values=past, use_cache=True)
        return out.logits[:, -1], (out.past_key_values, length + 1)

    # -- utilities --------------------------------------------------------
    def parameters(self):
        return self.model.parameters()

    def frozen_copy(self) -> Policy:
        """Deep copy of the model with gradients disabled (e.g. the KL reference)."""
        model = copy.deepcopy(self.model)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return Policy(self.backend, model, self.tokenizer, self.device, self.source)

    def save(self, directory: str | Path) -> None:
        """Write reloadable model and tokenizer files into an existing directory."""
        directory = Path(directory)
        if self.backend == "tiny":
            (directory / TINY_CONFIG_FILE).write_text(json.dumps(self.model.config, indent=2) + "\n", encoding="utf-8")
            (directory / TINY_TOKENIZER_FILE).write_text(
                json.dumps(self.tokenizer.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            torch.save(state, directory / TINY_WEIGHTS_FILE)
        else:
            self.model.save_pretrained(directory, safe_serialization=True)
            self.tokenizer.save_pretrained(directory)


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("backend 'hf' requires the optional 'hf' extra (the transformers package)") from exc
    return transformers


def load_policy(
    backend: str,
    model_name_or_path: str | None,
    device: str = "cpu",
    seed: int = 42,
) -> Policy:
    """Load a policy. ``tiny`` with no path creates a fresh seeded fixture model."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    torch_device = torch.device(device)
    if backend == "tiny":
        if model_name_or_path is None:
            torch.manual_seed(seed)
            model = TinyGRULM()
            source = f"fresh tiny GRU init (seed={seed})"
        else:
            path = Path(model_name_or_path)
            config_file = path / TINY_CONFIG_FILE
            if not config_file.is_file():
                raise FileNotFoundError(f"{path} is not a tiny checkpoint directory (missing {TINY_CONFIG_FILE})")
            config = json.loads(config_file.read_text(encoding="utf-8"))
            if config.get("architecture") != TINY_ARCHITECTURE:
                raise ValueError(f"{config_file} does not describe a {TINY_ARCHITECTURE} model")
            model = TinyGRULM(
                vocab_size=int(config["vocab_size"]),
                embedding_dim=int(config["embedding_dim"]),
                hidden_size=int(config["hidden_size"]),
                num_layers=int(config["num_layers"]),
            )
            state = torch.load(path / TINY_WEIGHTS_FILE, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            source = str(path)
        model.to(torch_device)
        return Policy("tiny", model, ByteTokenizer(), torch_device, source)

    if not model_name_or_path:
        raise ValueError("backend 'hf' requires model_name_or_path")
    local = Path(model_name_or_path)
    if local.is_dir() and (local / TINY_CONFIG_FILE).is_file():
        raise ValueError(f"{local} is a tiny checkpoint; use backend 'tiny' to load it")
    transformers = _import_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=False)
    model.to(torch_device)
    return Policy("hf", model, tokenizer, torch_device, str(model_name_or_path))


def completion_log_probs(
    policy: Policy, prompts: list[list[int]], completions: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of each completion given its prompt.

    Returns ``(log_probs, mask)`` of shape ``[N, T]`` with ``T`` the longest
    completion. Sequences are right-padded; labels are shifted by one so the
    token at position ``k`` is scored by the logits at ``k - 1``. Prompt and
    padding positions are excluded; completion tokens (including a sampled or
    supervised EOS) are included. Masked entries are exactly zero and receive
    no gradient.
    """
    if len(prompts) != len(completions) or not prompts:
        raise ValueError("prompts and completions must be non-empty lists of equal length")
    for prompt, completion in zip(prompts, completions, strict=True):
        if not prompt:
            raise ValueError("every prompt needs at least one token")
        if not completion:
            raise ValueError("every completion needs at least one token")
    lengths = [len(p) + len(c) for p, c in zip(prompts, completions, strict=True)]
    rows, width = len(prompts), max(lengths)
    input_ids = torch.full((rows, width), policy.pad_token_id, dtype=torch.long)
    attention = torch.zeros((rows, width), dtype=torch.long)
    for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        input_ids[i, : lengths[i]] = torch.tensor(prompt + completion, dtype=torch.long)
        attention[i, : lengths[i]] = 1
    input_ids = input_ids.to(policy.device)
    attention = attention.to(policy.device)

    logits = policy.forward_logits(input_ids[:, :-1], attention[:, :-1]).float()
    targets = input_ids[:, 1:]
    token_lp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)

    comp_lens = torch.tensor([len(c) for c in completions], device=policy.device)
    starts = torch.tensor([len(p) - 1 for p in prompts], device=policy.device)
    offsets = torch.arange(int(comp_lens.max()), device=policy.device)
    mask = offsets[None, :] < comp_lens[:, None]
    index = torch.where(mask, starts[:, None] + offsets[None, :], torch.zeros_like(offsets)[None, :])
    gathered = token_lp.gather(1, index)
    log_probs = torch.where(mask, gathered, torch.zeros_like(gathered))
    return log_probs, mask.to(log_probs.dtype)


def sample_next_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one token per row from raw full-vocabulary logits.

    Only temperature scaling and (if ``top_p < 1``) nucleus filtering are
    applied. ``top_p == 1`` keeps the full support, including special tokens.
    Sampling runs on CPU with the given CPU generator for reproducibility.
    """
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    probs = torch.softmax(logits.detach().float().cpu() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_probs = sorted_probs.masked_fill(mass_before >= top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    if not torch.isfinite(probs).all():
        raise FloatingPointError("non-finite sampling probabilities")
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def sample_completions(
    policy: Policy,
    prompt_ids: list[int],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> list[SampledCompletion]:
    """Sample ``num_samples`` completions of one prompt.

    Generation stops per row at EOS (which is kept as the final action). Rows
    that exhaust ``max_new_tokens`` are returned without an appended EOS and
    flagged ``stopped_on_eos=False``.
    """
    if num_samples < 1 or max_new_tokens < 1:
        raise ValueError("num_samples and max_new_tokens must be >= 1")
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    input_ids = torch.tensor([prompt_ids] * num_samples, dtype=torch.long, device=policy.device)
    logits, state = policy.start(input_ids)
    tokens: list[list[int]] = [[] for _ in range(num_samples)]
    done = [False] * num_samples
    for step in range(max_new_tokens):
        next_tokens = sample_next_tokens(logits, temperature, top_p, generator)
        for row, token in enumerate(next_tokens.tolist()):
            if done[row]:
                continue
            tokens[row].append(token)
            if token == policy.eos_token_id:
                done[row] = True
        if all(done) or step == max_new_tokens - 1:
            break
        logits, state = policy.advance(next_tokens.to(policy.device), state)

    completions = []
    for row_tokens, stopped in zip(tokens, done, strict=True):
        text_ids = row_tokens[:-1] if stopped else row_tokens
        completions.append(SampledCompletion(row_tokens, policy.decode(text_ids), stopped))
    return completions


def parameter_l2_distance(first: nn.Module, second: nn.Module) -> float:
    """L2 distance between two models with identical parameter layouts."""
    total = 0.0
    params_a = dict(first.named_parameters())
    params_b = dict(second.named_parameters())
    if params_a.keys() != params_b.keys():
        raise ValueError("models have different parameter names")
    for name, param in params_a.items():
        diff = param.detach().float().cpu() - params_b[name].detach().float().cpu()
        total += float(diff.pow(2).sum())
    return total**0.5
