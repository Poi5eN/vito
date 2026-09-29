#!/usr/bin/env python3
"""Deterministic VITO Startup Score v0.1 calculator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WEIGHTS = {
    "market_opportunity": 120,
    "product_problem_fit": 100,
    "traction": 150,
    "growth_momentum": 120,
    "retention_engagement": 100,
    "unit_economics_financial_quality": 120,
    "go_to_market_efficiency": 70,
    "technology_defensibility": 70,
    "team_execution": 80,
    "risk_resilience_data_quality": 70,
}

MISSING_SCORE = 50.0


def clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def score_startup(record: dict) -> dict:
    dimensions = record.get("dimensions", {})
    dimension_scores: dict[str, float] = {}
    evidence_coverage: dict[str, dict] = {}

    total = 0.0
    covered_weight = 0.0
    evidence_count = 0

    for name, weight in WEIGHTS.items():
        payload = dimensions.get(name, {})
        raw = payload.get("score")
        has_score = raw is not None
        score = clamp(raw if has_score else MISSING_SCORE)
        evidence_ids = payload.get("evidence_ids", []) or []

        dimension_scores[name] = score
        evidence_coverage[name] = {
            "weight": weight,
            "has_score": has_score,
            "evidence_count": len(evidence_ids),
        }

        total += weight * score / 100.0
        if has_score and evidence_ids:
            covered_weight += weight
            evidence_count += len(evidence_ids)

    coverage_pct = covered_weight / sum(WEIGHTS.values()) * 100.0
    provisional = coverage_pct < 90.0

    # v0.1 intentionally keeps the overall score at a full 0-1000 scale while
    # expressing evidence sufficiency separately as confidence/coverage.
    # This avoids treating missing information as proof of poor performance.
    confidence = coverage_pct

    return {
        "startup_id": record["startup_id"],
        "name": record["name"],
        "score": round(total, 2),
        "score_scale": 1000,
        "provisional": provisional,
        "confidence": round(confidence, 2),
        "as_of": record["as_of"],
        "methodology_version": "0.1.0",
        "dimension_scores": dimension_scores,
        "evidence_coverage": evidence_coverage,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <startup-record.json>")
        return 2

    path = Path(sys.argv[1])
    record = json.loads(path.read_text())
    result = score_startup(record)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
