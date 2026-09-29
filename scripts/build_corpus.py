#!/usr/bin/env python3
"""Build deterministic train/validation/test JSONL splits and a manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_for_tokenizer(record: dict) -> str:
    context = record.get("context", "").strip()
    return "\n".join([
        "<bos>",
        f"<domain>{record['domain']}</domain>",
        f"<task>{record['task']}</task>",
        f"<language>{record['language']}</language>",
        f"<input>{record['input'].strip()}</input>",
        f"<context>{context}</context>",
        f"<output>{record['expected_output'].strip()}</output>",
        "<eos>",
    ])


def load_records(path: Path) -> list[dict]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sid = record["sample_id"]
            if sid in seen:
                raise ValueError(f"duplicate sample_id: {sid}")
            seen.add(sid)
            records.append(record)
    return records


def deterministic_bucket(sample_id: str) -> float:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_split(records: list[dict]) -> dict[str, list[dict]]:
    # Stratify by domain for the seed set, while ensuring every split exists.
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["domain"]].append(record)

    splits = {"train": [], "validation": [], "test": []}
    for group in groups.values():
        group = sorted(group, key=lambda r: deterministic_bucket(r["sample_id"]))
        n = len(group)
        if n < 3:
            splits["train"].extend(group)
            continue

        test_n = max(1, round(n * 0.10))
        val_n = max(1, round(n * 0.10))
        if test_n + val_n >= n:
            test_n, val_n = 1, 1
        train_n = n - val_n - test_n

        splits["train"].extend(group[:train_n])
        splits["validation"].extend(group[train_n:train_n + val_n])
        splits["test"].extend(group[train_n + val_n:])

    for split in splits:
        splits[split].sort(key=lambda r: r["sample_id"])
    return splits


def stats(records: list[dict]) -> dict:
    chars = sum(len(r["input"]) + len(r.get("context", "")) + len(r["expected_output"]) for r in records)
    return {
        "records": len(records),
        "domains": dict(Counter(r["domain"] for r in records)),
        "tasks": dict(Counter(r["task"] for r in records)),
        "languages": dict(Counter(r["language"] for r in records)),
        "characters_in_training_text": chars,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/vito_corpus_v0.1"))
    args = parser.parse_args()

    records = load_records(args.input)
    splits = assign_split(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records in splits.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", split_records)
        text_path = args.output_dir / f"{split_name}.txt"
        with text_path.open("w", encoding="utf-8") as handle:
            for record in split_records:
                handle.write(text_for_tokenizer(record) + "\n")

    manifest = {
        "dataset_name": "vito-corpus",
        "version": "0.1.0",
        "seed": 42,
        "source_file": str(args.input),
        "files": {},
        "statistics": {
            "all": stats(records),
            "splits": {name: stats(items) for name, items in splits.items()},
        },
        "policy": {
            "benchmark_data_excluded": True,
            "seed_corpus_is_pipeline_only": True,
        },
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["statistics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
