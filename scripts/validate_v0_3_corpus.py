from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"][^'\"]+"
    ),
]


def contains_secret(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in SECRET_PATTERNS
    )


def validate_file(path: Path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "vito_corpus_v0.3"
        ),
    )

    args = parser.parse_args()

    print("=" * 64)
    print("VITO v0.3 CORPUS VALIDATION")
    print("=" * 64)

    all_ids = set()
    all_text_hashes = set()

    totals = Counter()
    problems = []

    for split in (
        "train",
        "validation",
        "test",
    ):
        path = (
            args.corpus_dir
            / f"{split}.jsonl"
        )

        records = validate_file(path)

        print(
            f"{split}: {len(records)} records"
        )

        totals[split] = len(records)

        for record in records:
            sample_id = record.get(
                "sample_id"
            )

            output = record.get(
                "output",
                "",
            )

            if not sample_id:
                problems.append(
                    "missing sample_id"
                )

            if not output.strip():
                problems.append(
                    f"{sample_id}: empty output"
                )

            if contains_secret(output):
                problems.append(
                    f"{sample_id}: possible secret"
                )

            if sample_id in all_ids:
                problems.append(
                    f"{sample_id}: duplicate ID"
                )

            all_ids.add(sample_id)

            text_hash = hash(
                output.strip()
            )

            if text_hash in all_text_hashes:
                problems.append(
                    f"{sample_id}: duplicate text"
                )

            all_text_hashes.add(text_hash)

            provenance = record.get(
                "provenance"
            )

            if not provenance:
                problems.append(
                    f"{sample_id}: missing provenance"
                )

    print()

    if problems:
        print(
            "VALIDATION FAILED"
        )

        for problem in problems[:50]:
            print(
                " -",
                problem,
            )

        print(
            "Problems:",
            len(problems),
        )

        raise SystemExit(1)

    print(
        "VALIDATION PASSED"
    )

    print(
        "Unique records:",
        len(all_ids),
    )

    print(
        "Splits:",
        dict(totals),
    )


if __name__ == "__main__":
    main()
