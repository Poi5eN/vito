from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_corpus_records_are_valid():
    mod = load_module(ROOT / "scripts" / "validate_corpus.py", "validate_corpus")
    path = ROOT / "data" / "seed" / "vito_seed_v0.1.jsonl"
    ids = set()
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        mod.validate_record(record, str(path), line_no)
        assert record["sample_id"] not in ids
        ids.add(record["sample_id"])
        count += 1
    assert count >= 20


def test_split_is_deterministic():
    mod = load_module(ROOT / "scripts" / "build_corpus.py", "build_corpus")
    records = [
        {"sample_id": f"vito_{i:06d}", "domain": "developer", "task": "debugging"}
        for i in range(1, 11)
    ]
    a = mod.assign_split(records)
    b = mod.assign_split(records)
    assert a == b
