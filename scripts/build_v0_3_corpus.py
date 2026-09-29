from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V02_SOURCE = (
    ROOT
    / "data"
    / "raw"
    / "vito_v0.2_expanded.jsonl"
)

EXTERNAL_SOURCE = (
    ROOT
    / "data"
    / "raw"
    / "external"
    / "cosmopedia_v0.3.jsonl"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def stable_hash(value: str) -> int:
    return int(
        sha256_text(value),
        16,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}"
        )

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
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc

            if isinstance(record, dict):
                records.append(record)

    return records


def normalize_vito(
    record: dict,
    index: int,
) -> dict:
    """
    Convert the existing VITO v0.2 schema into
    the unified v0.3 schema.

    Important:
    VITO v0.2 uses `expected_output`,
    not `output`.
    """

    output = str(
        record.get(
            "expected_output",
            "",
        )
    ).strip()

    sample_id = str(
        record.get(
            "sample_id",
            "",
        )
    ).strip()

    if not sample_id:
        sample_id = (
            f"vito_{index:06d}_"
            f"{sha256_text(output)[:16]}"
        )

    provenance = record.get(
        "provenance",
        {},
    )

    if not isinstance(
        provenance,
        dict,
    ):
        provenance = {}

    provenance = dict(provenance)

    provenance.setdefault(
        "source_type",
        "vito_project",
    )

    provenance.setdefault(
        "dataset",
        "vito_v0.2_expanded",
    )

    return {
        "sample_id": sample_id,
        "record_type": "structured",
        "domain": str(
            record.get(
                "domain",
                "unknown",
            )
        ),
        "task": str(
            record.get(
                "task",
                "unknown",
            )
        ),
        "language": str(
            record.get(
                "language",
                "en",
            )
        ),
        "input": str(
            record.get(
                "input",
                "",
            )
        ).strip(),
        "context": str(
            record.get(
                "context",
                "",
            )
        ).strip(),
        "output": output,
        "provenance": provenance,
        "verification": record.get(
            "verification",
            {},
        ),
        "metadata": record.get(
            "metadata",
            {},
        ),
    }


def normalize_external(
    record: dict,
    index: int,
) -> dict:
    """
    Normalize Cosmopedia into the unified
    VITO v0.3 text-record schema.
    """

    output = str(
        record.get(
            "output",
            "",
        )
    ).strip()

    sample_id = str(
        record.get(
            "sample_id",
            "",
        )
    ).strip()

    if not sample_id:
        sample_id = (
            f"cosmopedia_{index:06d}_"
            f"{sha256_text(output)[:16]}"
        )

    provenance = record.get(
        "provenance",
        {},
    )

    if not isinstance(
        provenance,
        dict,
    ):
        provenance = {}

    return {
        "sample_id": sample_id,
        "record_type": "text",
        "domain": "general",
        "task": "pretraining_text",
        "language": str(
            record.get(
                "language",
                "en",
            )
        ),
        "input": "",
        "context": "",
        "output": output,
        "provenance": provenance,
        "verification": record.get(
            "verification",
            {},
        ),
    }


def content_hash(
    record: dict,
) -> str:
    """
    Deterministic content fingerprint.

    Uses actual semantic content rather than
    sample_id so records with different IDs but
    identical content are still deduplicated.
    """

    fields = [
        "domain",
        "task",
        "language",
        "input",
        "context",
        "output",
    ]

    parts = []

    for field in fields:
        value = str(
            record.get(
                field,
                "",
            )
        ).strip()

        # Normalize whitespace.
        value = " ".join(
            value.split()
        )

        parts.append(value)

    normalized = "\n".join(parts)

    return sha256_text(
        normalized
    )


def deduplicate(
    records: list[dict],
) -> tuple[list[dict], int]:
    seen_content = set()
    seen_ids = set()

    unique_records = []
    duplicates_removed = 0

    for record in records:

        output = str(
            record.get(
                "output",
                "",
            )
        ).strip()

        # Empty records are invalid.
        if not output:
            continue

        record_id = str(
            record.get(
                "sample_id",
                "",
            )
        ).strip()

        if not record_id:
            record_id = (
                "record_"
                + sha256_text(
                    output
                )[:20]
            )

        # If an ID happens to collide, create a
        # deterministic unique ID. Do NOT discard
        # the record merely because its ID collides.
        original_id = record_id

        if record_id in seen_ids:
            suffix = sha256_text(
                content_hash(record)
            )[:8]

            record_id = (
                f"{original_id}_{suffix}"
            )

            counter = 2

            while record_id in seen_ids:
                record_id = (
                    f"{original_id}_{suffix}_{counter}"
                )
                counter += 1

        record["sample_id"] = record_id

        text_hash = content_hash(
            record
        )

        if text_hash in seen_content:
            duplicates_removed += 1
            continue

        seen_content.add(
            text_hash
        )

        seen_ids.add(
            record_id
        )

        unique_records.append(
            record
        )

    return (
        unique_records,
        duplicates_removed,
    )


def split_records(
    records: list[dict],
    seed: int,
):
    """
    Deterministic 90/5/5 split.
    """

    shuffled = list(records)

    shuffled.sort(
        key=lambda record: stable_hash(
            f"{seed}:{record['sample_id']}"
        )
    )

    total = len(shuffled)

    train_end = int(
        total * 0.90
    )

    validation_end = int(
        total * 0.95
    )

    train = shuffled[
        :train_end
    ]

    validation = shuffled[
        train_end:validation_end
    ]

    test = shuffled[
        validation_end:
    ]

    return (
        train,
        validation,
        test,
    )


def format_record(
    record: dict,
) -> str:
    """
    Convert a normalized record into the
    training text representation.
    """

    if record.get(
        "record_type"
    ) == "text":

        return (
            "<bos>\n"
            "<domain>general</domain>\n"
            "<task>pretraining_text</task>\n"
            "<language>en</language>\n"
            "<output>\n"
            f"{record['output']}\n"
            "</output>\n"
            "<eos>\n"
        )

    return (
        "<bos>\n"
        f"<domain>{record.get('domain', '')}</domain>\n"
        f"<task>{record.get('task', '')}</task>\n"
        f"<language>{record.get('language', 'en')}</language>\n"
        f"<input>{record.get('input', '')}</input>\n"
        f"<context>{record.get('context', '')}</context>\n"
        f"<output>{record.get('output', '')}</output>\n"
        "<eos>\n"
    )


def write_split(
    records: list[dict],
    output_dir: Path,
    name: str,
):
    jsonl_path = (
        output_dir
        / f"{name}.jsonl"
    )

    text_path = (
        output_dir
        / f"{name}.txt"
    )

    with (
        jsonl_path.open(
            "w",
            encoding="utf-8",
        ) as jsonl_handle,
        text_path.open(
            "w",
            encoding="utf-8",
        ) as text_handle,
    ):
        for record in records:

            jsonl_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            text_handle.write(
                format_record(
                    record
                )
            )


def count_domains(
    records: list[dict],
) -> dict:
    return dict(
        Counter(
            record.get(
                "domain",
                "unknown",
            )
            for record in records
        )
    )


def count_sources(
    records: list[dict],
) -> dict:
    return dict(
        Counter(
            record.get(
                "provenance",
                {},
            ).get(
                "source_type",
                "unknown",
            )
            for record in records
        )
    )


def count_record_types(
    records: list[dict],
) -> dict:
    return dict(
        Counter(
            record.get(
                "record_type",
                "unknown",
            )
            for record in records
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the VITO v0.3 unified corpus."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "vito_corpus_v0.3"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("=" * 64)
    print("BUILDING VITO v0.3 CORPUS")
    print("=" * 64)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    print()
    print("Loading sources...")
    print()

    raw_vito = read_jsonl(
        V02_SOURCE
    )

    raw_external = read_jsonl(
        EXTERNAL_SOURCE
    )

    print(
        "VITO records:",
        len(raw_vito),
    )

    print(
        "Cosmopedia records:",
        len(raw_external),
    )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    vito_records = [
        normalize_vito(
            record,
            index,
        )
        for index, record in enumerate(
            raw_vito
        )
    ]

    external_records = [
        normalize_external(
            record,
            index,
        )
        for index, record in enumerate(
            raw_external
        )
    ]

    # ---------------------------------------------------------
    # Validate source conversion
    # ---------------------------------------------------------

    empty_vito = [
        record
        for record in vito_records
        if not record["output"]
    ]

    empty_external = [
        record
        for record in external_records
        if not record["output"]
    ]

    if empty_vito:
        raise RuntimeError(
            f"{len(empty_vito)} VITO records "
            "have empty output after normalization."
        )

    if empty_external:
        raise RuntimeError(
            f"{len(empty_external)} Cosmopedia records "
            "have empty output after normalization."
        )

    print()
    print("Normalization")
    print("-" * 64)

    print(
        "VITO normalized:",
        len(vito_records),
    )

    print(
        "Cosmopedia normalized:",
        len(external_records),
    )

    print(
        "VITO domains:",
        count_domains(vito_records),
    )

    print(
        "External domains:",
        count_domains(external_records),
    )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    all_records = (
        vito_records
        + external_records
    )

    print()
    print("Before deduplication")
    print("-" * 64)

    print(
        "Total:",
        len(all_records),
    )

    print(
        "Domains:",
        count_domains(all_records),
    )

    print(
        "Sources:",
        count_sources(all_records),
    )

    # ---------------------------------------------------------
    # Deduplicate
    # ---------------------------------------------------------

    records, duplicates_removed = (
        deduplicate(
            all_records
        )
    )

    print()
    print("After deduplication")
    print("-" * 64)

    print(
        "Unique records:",
        len(records),
    )

    print(
        "Duplicates removed:",
        duplicates_removed,
    )

    print(
        "Domains:",
        count_domains(records),
    )

    print(
        "Sources:",
        count_sources(records),
    )

    print(
        "Record types:",
        count_record_types(records),
    )

    # ---------------------------------------------------------
    # Source preservation
    # ---------------------------------------------------------

    vito_preserved = sum(
        1
        for record in records
        if record.get(
            "provenance",
            {},
        ).get(
            "source_type"
        ) == "vito_project"
    )

    cosmopedia_preserved = sum(
        1
        for record in records
        if record.get(
            "provenance",
            {},
        ).get(
            "source_type"
        ) == "huggingface"
    )

    print()
    print("Source preservation")
    print("-" * 64)

    print(
        f"VITO:        "
        f"{vito_preserved} / "
        f"{len(vito_records)}"
    )

    print(
        f"Cosmopedia:  "
        f"{cosmopedia_preserved} / "
        f"{len(external_records)}"
    )

    if vito_preserved == 0:
        raise RuntimeError(
            "No VITO records survived. "
            "Aborting before writing the corpus."
        )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    train, validation, test = (
        split_records(
            records,
            args.seed,
        )
    )

    # ---------------------------------------------------------
    # Write
    # ---------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_split(
        train,
        args.output_dir,
        "train",
    )

    write_split(
        validation,
        args.output_dir,
        "validation",
    )

    write_split(
        test,
        args.output_dir,
        "test",
    )

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    manifest = {
        "version": "0.3.0",
        "seed": args.seed,

        "total_records": len(records),

        "duplicates_removed": (
            duplicates_removed
        ),

        "splits": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },

        "domains": count_domains(
            records
        ),

        "record_types": count_record_types(
            records
        ),

        "source_types": count_sources(
            records
        ),

        "source_counts": {
            "vito_project": vito_preserved,
            "huggingface": cosmopedia_preserved,
        },

        "source_files": {
            "vito": str(V02_SOURCE),
            "cosmopedia": str(
                EXTERNAL_SOURCE
            ),
        },
    }

    manifest_path = (
        args.output_dir
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("VITO v0.3 CORPUS COMPLETE")
    print("=" * 64)

    print(
        "Total:",
        len(records),
    )

    print(
        "Train:",
        len(train),
    )

    print(
        "Validation:",
        len(validation),
    )

    print(
        "Test:",
        len(test),
    )

    print()
    print("Domains:")

    for domain, count in sorted(
        count_domains(
            records
        ).items()
    ):
        print(
            f"  {domain}: {count}"
        )

    print()
    print("Sources:")

    for source, count in sorted(
        count_sources(
            records
        ).items()
    ):
        print(
            f"  {source}: {count}"
        )

    print()
    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Output:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
