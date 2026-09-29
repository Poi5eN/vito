#!/usr/bin/env python3
"""Validate VITO corpus JSONL records without external dependencies."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ID_RE = re.compile(r"^vito_\d{6}$")
REQUIRED = [
    "sample_id", "version", "domain", "task", "language",
    "input", "expected_output", "verification", "evidence", "provenance"
]


def fail(path: str, line_no: int, message: str) -> None:
    raise ValueError(f"{path}:{line_no}: {message}")


def validate_record(record: dict, path: str, line_no: int) -> None:
    missing = [key for key in REQUIRED if key not in record]
    if missing:
        fail(path, line_no, f"missing fields: {missing}")
    if not ID_RE.fullmatch(record["sample_id"]):
        fail(path, line_no, "sample_id must match vito_000000 format")
    if record["domain"] not in {"developer", "venture"}:
        fail(path, line_no, "invalid domain")
    for field in ("input", "expected_output"):
        if not isinstance(record[field], str) or not record[field].strip():
            fail(path, line_no, f"{field} must be a non-empty string")
    if not isinstance(record["verification"], list) or not record["verification"]:
        fail(path, line_no, "verification must be a non-empty list")
    if not isinstance(record["evidence"], list):
        fail(path, line_no, "evidence must be a list")
    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        fail(path, line_no, "provenance must be an object")
    for key in ("source_type", "license", "creator"):
        if not str(provenance.get(key, "")).strip():
            fail(path, line_no, f"provenance.{key} must be non-empty")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_corpus.py <file.jsonl>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    count = 0
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                fail(str(path), line_no, f"invalid JSON: {exc}")
            if not isinstance(record, dict):
                fail(str(path), line_no, "record must be an object")
            validate_record(record, str(path), line_no)
            sample_id = record["sample_id"]
            if sample_id in ids:
                fail(str(path), line_no, f"duplicate sample_id: {sample_id}")
            ids.add(sample_id)
            count += 1

    print(f"Validated {count} VITO records: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
