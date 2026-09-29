"""JSONL datasets and schema validation.

Schemas (extra keys are preserved as metadata):

* RL examples: ``prompt: str`` plus ``answer: str`` for ``domain="reasoning"`` (the
  default; optional with ``require_answer=False`` for verifier-scored or unlabeled data);
  ``domain="general"`` examples need no answer and are scored by preference callbacks.
* SFT rows: ``prompt: str`` and ``response: str``.
* Candidate rows (sampling output, rejection/evaluation input): ``prompt: str``,
  ``answer: str`` and ``responses: list[str]``.

All kinds accept an optional ``id`` (string or integer) and ``domain``
(``"reasoning"`` or ``"general"``).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DOMAINS = ("reasoning", "general")

__all__ = ["DOMAINS", "merge_sft", "read_jsonl", "validate_candidates", "validate_examples", "write_jsonl"]


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    """Read a JSONL file of objects; blank lines are skipped."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object, got {type(record).__name__}")
            records.append(record)
    return records


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict]) -> None:
    """Write records as UTF-8 JSONL, creating parent directories. Writes atomically."""
    path = Path(path)
    lines = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record {index} must be a dict, got {type(record).__name__}")
        try:
            lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {index} is not JSON serializable: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for line in lines:
                stream.write(line + "\n")
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _require_records(records: Any) -> None:
    if not isinstance(records, list):
        raise TypeError(f"records must be a list of dicts, got {type(records).__name__}")
    if not records:
        raise ValueError("dataset is empty")


def _require_text(record: dict, key: str, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"record {index}: field {key!r} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"record {index}: field {key!r} must not be empty")
    return value


def _check_common(record: Any, index: int) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"record {index} must be a dict, got {type(record).__name__}")
    _require_text(record, "prompt", index)
    if "id" in record:
        record_id = record["id"]
        if isinstance(record_id, bool) or not isinstance(record_id, (str, int)):
            raise ValueError(f"record {index}: field 'id' must be a string or integer")
    domain = record.get("domain", "reasoning")
    if domain not in DOMAINS:
        raise ValueError(f"record {index}: field 'domain' must be one of {DOMAINS}, got {domain!r}")
    return domain


def validate_examples(records: list[dict], supervised: bool = False, *, require_answer: bool = True) -> None:
    """Validate RL examples (``supervised=False``) or SFT rows (``supervised=True``).

    ``require_answer=False`` lets reasoning RL rows omit ``answer`` (verifier-scored code
    datasets, unlabeled sampling); an ``answer`` that is present must still be a non-empty
    string. The flag does not affect SFT rows or general-domain rows.
    """
    _require_records(records)
    for index, record in enumerate(records):
        domain = _check_common(record, index)
        if supervised:
            _require_text(record, "response", index)
        elif domain == "reasoning":
            if require_answer or "answer" in record:
                _require_text(record, "answer", index)
        elif "answer" in record and not isinstance(record["answer"], str):
            raise ValueError(f"record {index}: field 'answer' must be a string")


def validate_candidates(records: list[dict], *, require_answer: bool = True) -> None:
    """Validate sampled candidate rows (reasoning domain only)."""
    _require_records(records)
    for index, record in enumerate(records):
        domain = _check_common(record, index)
        if domain != "reasoning":
            raise ValueError(f"record {index}: candidates must be reasoning-domain rows, got {domain!r}")
        if require_answer:
            _require_text(record, "answer", index)
        elif "answer" in record and not isinstance(record["answer"], str):
            raise ValueError(f"record {index}: field 'answer' must be a string")
        responses = record.get("responses")
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"record {index}: field 'responses' must be a non-empty list of strings")
        if not all(isinstance(response, str) for response in responses):
            raise ValueError(f"record {index}: every entry of 'responses' must be a string")


def merge_sft(reasoning: list[dict], general: list[dict]) -> list[dict]:
    """Combine reasoning and general SFT rows, tagging ``domain`` and ``source`` provenance.

    Rows are copied, never invented; an existing ``source`` field (a non-empty string) is
    kept. Both inputs must be non-empty: mixed-data SFT without one of the two sources is
    rejected rather than silently degraded. Order is all reasoning rows followed by all
    general rows (trainers shuffle).
    """
    for name, rows in (("reasoning", reasoning), ("general", general)):
        try:
            validate_examples(rows, supervised=True)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{name} SFT rows: {exc}") from exc
    merged: list[dict] = []
    for name, rows in (("reasoning", reasoning), ("general", general)):
        for index, row in enumerate(rows):
            if row.get("domain", name) != name:
                raise ValueError(f"{name} row {index} is labeled domain {row['domain']!r}")
            source = row.get("source", name)
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{name} row {index}: field 'source' must be a non-empty string")
            combined = dict(row)
            combined["domain"] = name
            combined.setdefault("source", name)
            merged.append(combined)
    return merged
