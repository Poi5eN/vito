#!/usr/bin/env python3

"""
VITO Dataset Quality Gate

Validates the processed VITO v0.3 corpus without imposing
external-source requirements on VITO-native records.

Checks:
- JSON validity
- required normalized fields
- empty content
- provenance
- verification metadata
- duplicate sample IDs
- duplicate full records
- cross-split leakage
- secret-like patterns
- corpus statistics
- approximate token count
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CORPUS_DIR = Path("data/processed/vito_corpus_v0.3")

SPLITS = ("train", "validation", "test")

REQUIRED_FIELDS = {
    "sample_id",
    "domain",
    "task",
    "language",
    "input",
    "context",
    "output",
    "provenance",
    "verification",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
]


def fingerprint(record: dict) -> str:
    """Create a stable fingerprint of the complete normalized record."""

    payload = {
        "domain": record.get("domain"),
        "task": record.get("task"),
        "language": record.get("language"),
        "input": record.get("input"),
        "context": record.get("context"),
        "output": record.get("output"),
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def approx_tokens(text: str) -> int:
    """
    Rough token estimate.

    This is intentionally approximate.
    Exact tokenizer statistics will be calculated later
    using the VITO tokenizer.
    """

    return max(1, len(text) // 4)


def load_split(split: str) -> list[dict]:
    path = CORPUS_DIR / f"{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{split}:{line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{split}:{line_number}: record is not an object"
                )

            record["_line_number"] = line_number
            records.append(record)

    return records


def main() -> None:

    print("=" * 68)
    print("VITO DATASET QUALITY GATE")
    print("=" * 68)
    print()

    errors: list[str] = []

    split_records: dict[str, list[dict]] = {}

    for split in SPLITS:

        try:
            records = load_split(split)
        except Exception as exc:
            errors.append(str(exc))
            records = []

        split_records[split] = records

        print(f"{split:<12}: {len(records):>7} records")

    print()
    print("-" * 68)
    print("DATASET SUMMARY")
    print("-" * 68)

    all_records: list[dict] = []

    for split in SPLITS:
        all_records.extend(split_records[split])

    print(f"Total records: {len(all_records):>12}")

    total_characters = 0
    total_approx_tokens = 0

    domain_counts = Counter()
    source_counts = Counter()
    record_type_counts = Counter()

    sample_id_locations = defaultdict(list)
    content_locations = defaultdict(list)

    # ------------------------------------------------------------
    # Record-level validation
    # ------------------------------------------------------------

    for split in SPLITS:

        for record in split_records[split]:

            line_number = record.get("_line_number", "?")
            location = f"{split}:{line_number}"

            # ----------------------------------------
            # Required normalized fields
            # ----------------------------------------

            missing_fields = REQUIRED_FIELDS - set(record.keys())

            for field in sorted(missing_fields):
                errors.append(
                    f"{location}: missing required field '{field}'"
                )

            # ----------------------------------------
            # Sample ID
            # ----------------------------------------

            sample_id = record.get("sample_id")

            if not sample_id:
                errors.append(
                    f"{location}: empty sample_id"
                )
            else:
                sample_id_locations[str(sample_id)].append(location)

            # ----------------------------------------
            # Content
            # ----------------------------------------

            input_text = str(record.get("input") or "")
            context_text = str(record.get("context") or "")
            output_text = str(record.get("output") or "")

            combined_content = "\n".join(
                part
                for part in (
                    input_text,
                    context_text,
                    output_text,
                )
                if part.strip()
            )

            # IMPORTANT:
            #
            # Structured VITO records may legitimately have an
            # empty `input` field because the actual task payload
            # lives in `context`.
            #
            # Therefore we require meaningful record content,
            # not specifically a non-empty `input`.

            if not combined_content.strip():
                errors.append(
                    f"{location}: empty record content"
                )

            if not output_text.strip():
                errors.append(
                    f"{location}: empty output"
                )

            # ----------------------------------------
            # Statistics
            # ----------------------------------------

            total_characters += len(combined_content)
            total_approx_tokens += approx_tokens(combined_content)

            domain_counts[
                str(record.get("domain") or "unknown")
            ] += 1

            record_type_counts[
                str(record.get("record_type") or "structured")
            ] += 1

            # ----------------------------------------
            # Provenance
            # ----------------------------------------

            provenance = record.get("provenance")

            if not isinstance(provenance, dict):
                errors.append(
                    f"{location}: provenance is not an object"
                )
            else:

                # VITO-native records use `source`.
                # External records use `source_type`.
                source = (
                    provenance.get("source")
                    or provenance.get("source_type")
                )

                if not source:
                    errors.append(
                        f"{location}: provenance missing source"
                    )

                # External records should have a license.
                # VITO-native records may use project-owned
                # licensing metadata.
                if not provenance.get("license"):
                    errors.append(
                        f"{location}: provenance missing license"
                    )

            # ----------------------------------------
            # Verification
            # ----------------------------------------

            verification = record.get("verification")

            if not isinstance(verification, dict):
                errors.append(
                    f"{location}: verification is not an object"
                )

            else:

                method = verification.get("method")

                if not method:
                    errors.append(
                        f"{location}: verification missing method"
                    )

                # VITO-native bootstrap records use:
                #
                #   required: true
                #   method: ...
                #
                # External records use:
                #
                #   status: ...
                #   method: ...
                #
                # Both are valid representations under the
                # current v0.3 normalization policy.

                source_type = (
                    (record.get("provenance") or {})
                    .get("source_type")
                )

                if source_type == "vito_project":

                    if verification.get("required") is not True:
                        errors.append(
                            f"{location}: "
                            "VITO verification.required != true"
                        )

                else:

                    if not verification.get("status"):
                        errors.append(
                            f"{location}: "
                            "external verification missing status"
                        )

            # ----------------------------------------
            # Duplicate full content
            # ----------------------------------------

            content_locations[
                fingerprint(record)
            ].append(location)

            # ----------------------------------------
            # Secret detection
            # ----------------------------------------

            serialized = json.dumps(
                record,
                ensure_ascii=False,
            )

            for pattern in SECRET_PATTERNS:

                if pattern.search(serialized):
                    errors.append(
                        f"{location}: possible secret detected"
                    )
                    break

    # ------------------------------------------------------------
    # Duplicate sample IDs
    # ------------------------------------------------------------

    for sample_id, locations in sample_id_locations.items():

        if len(locations) > 1:
            errors.append(
                f"duplicate sample_id '{sample_id}': "
                + ", ".join(locations)
            )

    # ------------------------------------------------------------
    # Duplicate full records
    # ------------------------------------------------------------

    duplicate_groups = 0
    duplicate_records = 0

    for locations in content_locations.values():

        if len(locations) > 1:

            duplicate_groups += 1
            duplicate_records += len(locations)

            errors.append(
                "duplicate content: "
                + ", ".join(locations)
            )

    # ------------------------------------------------------------
    # Cross-split leakage
    # ------------------------------------------------------------

    fingerprints_by_split = {}

    for split in SPLITS:

        fingerprints_by_split[split] = {
            fingerprint(record)
            for record in split_records[split]
        }

    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for left, right in split_pairs:

        overlap = (
            fingerprints_by_split[left]
            & fingerprints_by_split[right]
        )

        if overlap:

            errors.append(
                f"cross-split leakage: {left} vs {right}: "
                f"{len(overlap)} overlapping records"
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print(f"Characters: {total_characters:>17,}")
    print(f"Approx tokens: {total_approx_tokens:>14,}")

    print()
    print("Domains:")

    for key, value in sorted(domain_counts.items()):
        print(f"  {key:<24} {value}")

    print()
    print("Sources:")

    for split in SPLITS:

        for record in split_records[split]:

            provenance = record.get("provenance") or {}

            source = (
                provenance.get("source_type")
                or provenance.get("source")
                or "unknown"
            )

            source_counts[str(source)] += 1

    for key, value in sorted(source_counts.items()):
        print(f"  {key:<24} {value}")

    print()
    print("Record types:")

    for key, value in sorted(record_type_counts.items()):
        print(f"  {key:<24} {value}")

    print()
    print("-" * 68)

    print(
        f"Duplicate content groups: {duplicate_groups}"
    )

    print(
        f"Records involved in duplicate content: "
        f"{duplicate_records}"
    )

    print()
    print("-" * 68)

    if errors:

        print(
            f"QUALITY GATE: FAILED "
            f"({len(errors)} problems)"
        )

        print("-" * 68)

        # Avoid flooding the terminal with thousands of errors.
        max_display = 100

        for error in errors[:max_display]:
            print(f"ERROR: {error}")

        if len(errors) > max_display:
            print()
            print(
                f"... and {len(errors) - max_display} more problems"
            )

        raise SystemExit(1)

    print("QUALITY GATE: PASSED")
    print()
    print("All dataset integrity checks passed.")
    print("=" * 68)


if __name__ == "__main__":
    main()