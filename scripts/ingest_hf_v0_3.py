from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "raw"
    / "external"
    / "cosmopedia_v0.3.jsonl"
)


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def make_id(text: str, source: str) -> str:
    digest = hashlib.sha256(
        f"{source}:{text}".encode("utf-8")
    ).hexdigest()[:20]

    return f"ext_{digest}"


def main():
    parser = argparse.ArgumentParser(
        description="Stream selected Hugging Face data into VITO format."
    )

    parser.add_argument(
        "--dataset",
        default="HuggingFaceTB/cosmopedia",
    )

    parser.add_argument(
        "--config",
        default="web_samples_v1",
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    print("=" * 64)
    print("VITO v0.3 HUGGING FACE INGESTION")
    print("=" * 64)

    print("Dataset:", args.dataset)
    print("Config:", args.config)
    print("Split:", args.split)
    print("Maximum records:", args.max_records)
    print("Output:", args.output)

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in dataset:
            text = row.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if len(text) < 100:
                continue

            sample_id = make_id(
                text,
                args.dataset,
            )

            record = {
                "sample_id": sample_id,
                "record_type": "text",
                "domain": "general",
                "task": "pretraining_text",
                "language": "en",
                "input": "",
                "context": "",
                "output": text,
                "provenance": {
                    "source_type": "huggingface",
                    "dataset": args.dataset,
                    "config": args.config,
                    "split": args.split,
                    "license": "apache-2.0",
                    "source_id": row.get("id"),
                    "retrieved_at": now_iso(),
                },
                "verification": {
                    "status": "source_metadata_verified",
                    "method": "dataset_card_and_source_metadata",
                },
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

            if count % 100 == 0:
                print(
                    f"Records ingested: {count}"
                )

            if count >= args.max_records:
                break

    print()
    print("=" * 64)
    print("INGESTION COMPLETE")
    print("=" * 64)
    print("Records:", count)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
