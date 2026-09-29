"""Command-line interface: ``r1-grpo`` / ``python -m r1_grpo``.

Every heavy dependency (torch, transformers) is imported lazily inside the
command handlers so that ``r1-grpo --help`` stays fast.

All file paths given on the command line or inside JSON configs are resolved
relative to the current working directory. The current working directory is
also appended to ``sys.path`` so that callback specs such as
``examples.callbacks:toy_helpfulness`` resolve when run from the repository root.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from r1_grpo import __version__

PROG = "r1-grpo"


class CLIError(ValueError):
    """An invalid user input that should be reported without a traceback."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON config object from ``path`` (relative to the working directory)."""
    config_path = Path(path)
    if not config_path.is_file():
        raise CLIError(
            f"config file not found: {str(path)!r} "
            f"(paths are resolved relative to the current directory {os.getcwd()!r})"
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"config {str(path)!r} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        kind = type(data).__name__
        raise CLIError(f"config {str(path)!r} must contain a JSON object, got {kind}")
    return data


def load_callable(spec: str) -> Callable[..., Any]:
    """Resolve a ``'package.module:function'`` spec to a callable."""
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise CLIError(f"callback spec must look like 'module:function', got {spec!r}")
    module_name, attr_path = (part.strip() for part in spec.split(":"))
    if not module_name or not attr_path:
        raise CLIError(f"callback spec must look like 'module:function', got {spec!r}")
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise CLIError(f"cannot import callback module {module_name!r}: {exc}") from exc
    for attr in attr_path.split("."):
        try:
            target = getattr(target, attr)
        except AttributeError as exc:
            raise CLIError(f"callback {spec!r}: {module_name!r} has no attribute {attr_path!r}") from exc
    if not callable(target):
        raise CLIError(f"callback {spec!r} is not callable")
    return target


def _ensure_cwd_importable() -> None:
    cwd = os.getcwd()
    if cwd not in sys.path and "" not in sys.path:
        sys.path.append(cwd)


def _read_jsonl_allowing_empty(path: str) -> list[dict[str, Any]]:
    """Read JSONL, returning ``[]`` for an existing file with no records.

    An empty rejection-sampling output is a legitimate result (e.g. for an
    untrained model); ``mix`` detects it to report a specific error.
    """
    from r1_grpo.data import read_jsonl

    file_path = Path(path)
    if file_path.is_file() and not file_path.read_text(encoding="utf-8").strip():
        return []
    return read_jsonl(path)


def _require_file(path: str, what: str) -> None:
    if not Path(path).is_file():
        raise CLIError(
            f"{what} not found: {path!r} (paths are resolved relative to the current directory {os.getcwd()!r})"
        )


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False))


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_sft(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from r1_grpo.training import train_sft

    _emit(train_sft(config))
    return 0


def _cmd_grpo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from r1_grpo.training import train_grpo

    _emit(train_grpo(config))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from r1_grpo.data import write_jsonl
    from r1_grpo.training import generate_candidates

    records = generate_candidates(config)
    write_jsonl(args.output, records)
    _emit(
        {
            "output": args.output,
            "prompts": len(records),
            "responses": sum(len(r.get("responses", [])) for r in records),
        }
    )
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    _require_file(args.input, "candidate file")
    verifier = load_callable(args.verifier) if args.verifier else None
    language_scorer = load_callable(args.language_scorer) if args.language_scorer else None

    from r1_grpo.data import read_jsonl, write_jsonl
    from r1_grpo.rejection import filter_candidates

    records = read_jsonl(args.input)
    accepted = filter_candidates(
        records,
        min_language_consistency=args.min_language_consistency,
        max_paragraph_chars=args.max_paragraph_chars,
        verifier=verifier,
        language_scorer=language_scorer,
    )
    write_jsonl(args.output, accepted)
    if not accepted:
        print(
            f"{PROG} reject: warning: no responses were accepted; {args.output!r} is empty. "
            "This is expected for an untrained model; 'mix' and 'sft' reject empty data.",
            file=sys.stderr,
        )
    _emit(
        {
            "output": args.output,
            "prompts": len(records),
            "candidate_responses": sum(len(r.get("responses", [])) for r in records),
            "accepted": len(accepted),
        }
    )
    return 0


def _cmd_mix(args: argparse.Namespace) -> int:
    _require_file(args.reasoning, "reasoning SFT file")
    _require_file(args.general, "general SFT file")
    from r1_grpo.data import merge_sft, write_jsonl

    reasoning = _read_jsonl_allowing_empty(args.reasoning)
    if not reasoning:
        raise CLIError(
            f"reasoning SFT file {args.reasoning!r} is empty: rejection sampling accepted no "
            "responses, so there is no reasoning data to mix (expected for an untrained model; "
            "see README)"
        )
    general = _read_jsonl_allowing_empty(args.general)
    merged = merge_sft(reasoning, general)
    write_jsonl(args.output, merged)
    _emit(
        {
            "output": args.output,
            "reasoning": len(reasoning),
            "general": len(general),
            "total": len(merged),
        }
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    _require_file(args.input, "candidate file")
    from r1_grpo.data import read_jsonl
    from r1_grpo.evaluation import evaluate_records

    verifier = load_callable(args.verifier) if args.verifier else None
    report = evaluate_records(read_jsonl(args.input), verifier=verifier)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(report)
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from r1_grpo.training import smoke

    _emit(smoke(args.output_dir))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Study implementation of the DeepSeek-R1 (v1) pipeline: R1-Zero GRPO, cold-start "
            "SFT, reasoning RL, rejection sampling + mixed SFT, all-scenario RL, and SFT-only "
            "distillation. Paths are relative to the current working directory."
        ),
        epilog=("Run '%(prog)s COMMAND --help' for command options. See README.md for the full pipeline."),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--traceback",
        action="store_true",
        help="show the full Python traceback on errors (for debugging)",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser(
        "sft",
        parents=[common],
        help="supervised fine-tuning (stage cold_start, mixed_sft or distill)",
        description=("Supervised fine-tuning. The config 'stage' must be cold_start, mixed_sft or distill."),
    )
    p.add_argument("--config", required=True, help="JSON config file")
    p.set_defaults(handler=_cmd_sft)

    p = sub.add_parser(
        "grpo",
        parents=[common],
        help="GRPO reinforcement learning (stage zero, reasoning or all)",
        description="GRPO training. The config 'stage' must be zero, reasoning or all.",
    )
    p.add_argument("--config", required=True, help="JSON config file")
    p.set_defaults(handler=_cmd_grpo)

    p = sub.add_parser(
        "sample",
        parents=[common],
        help="sample candidate responses (for rejection sampling or evaluation)",
        description=(
            "Sample num_samples responses per prompt from a checkpoint and write candidate "
            "records {prompt, answer, responses, ...} as JSONL."
        ),
    )
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--output", required=True, help="output candidate JSONL file")
    p.set_defaults(handler=_cmd_sample)

    p = sub.add_parser(
        "reject",
        parents=[common],
        help="rejection-sample candidates into reasoning SFT rows",
        description=(
            "Keep only correct, well-structured, readable responses and write SFT rows "
            "{prompt, response, domain='reasoning'}."
        ),
    )
    p.add_argument("--input", required=True, help="candidate JSONL from 'sample'")
    p.add_argument("--output", required=True, help="output SFT JSONL file")
    p.add_argument(
        "--min-language-consistency",
        type=_fraction,
        default=0.9,
        help="minimum language-consistency score of the reasoning, in [0, 1] (default: 0.9)",
    )
    p.add_argument(
        "--max-paragraph-chars",
        type=_positive_int,
        default=2000,
        help="reject reasoning containing a longer paragraph (default: 2000)",
    )
    p.add_argument("--verifier", help="optional 'module:function' correctness verifier")
    p.add_argument("--language-scorer", help="optional 'module:function' language scorer")
    p.set_defaults(handler=_cmd_reject)

    p = sub.add_parser(
        "mix",
        parents=[common],
        help="merge reasoning and general SFT data",
        description="Merge reasoning SFT rows with general (non-reasoning) SFT rows.",
    )
    p.add_argument("--reasoning", required=True, help="reasoning SFT JSONL (from 'reject')")
    p.add_argument("--general", required=True, help="general SFT JSONL")
    p.add_argument("--output", required=True, help="output mixed SFT JSONL")
    p.set_defaults(handler=_cmd_mix)

    p = sub.add_parser(
        "evaluate",
        parents=[common],
        help="score candidates: averaged pass@1 and majority consensus",
        description=(
            "Compute pass@1 averaged over the k samples of each question (then over questions) "
            "and normalized-answer majority-vote consensus. Not an any-correct metric."
        ),
    )
    p.add_argument("--input", required=True, help="candidate JSONL from 'sample'")
    p.add_argument("--output", help="optional JSON report file")
    p.add_argument("--verifier", help="optional 'module:function' correctness verifier")
    p.set_defaults(handler=_cmd_evaluate)

    p = sub.add_parser(
        "smoke",
        parents=[common],
        help="offline CPU end-to-end check with the tiny test fixture model",
        description=(
            "Tiny SFT, on-policy R1-Zero GRPO, sampling, evaluation and checkpoint reload with "
            "the offline tiny fixture. Verifies plumbing only; it does not learn reasoning."
        ),
    )
    p.add_argument("--output-dir", required=True, help="directory for smoke outputs")
    p.set_defaults(handler=_cmd_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_cwd_importable()
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print(f"{PROG} {args.command}: interrupted", file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError, ArithmeticError) as exc:
        if args.traceback:
            traceback.print_exc()
        message = str(exc) if isinstance(exc, CLIError) else f"{type(exc).__name__}: {exc}"
        print(f"{PROG} {args.command}: error: {message}", file=sys.stderr)
        if not args.traceback:
            print("(re-run with --traceback for details)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
