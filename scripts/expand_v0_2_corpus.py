#!/usr/bin/env python3

"""
Expand the VITO v0.2 bootstrap corpus with deterministic
domain-specific synthetic scenarios.

These records are explicitly marked synthetic and project-generated.
They are bootstrap training data, not a substitute for licensed
external datasets.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "raw" / "vito_v0.2_records.jsonl"
OUTPUT = ROOT / "data" / "raw" / "vito_v0.2_expanded.jsonl"


DEVELOPER_SCENARIOS = [
    ("Node.js", "API", "HTTP 500"),
    ("Node.js", "database", "connection timeout"),
    ("Node.js", "memory", "memory growth"),
    ("Python", "API", "HTTP 500"),
    ("Python", "database", "connection timeout"),
    ("Python", "worker", "high CPU usage"),
    ("PostgreSQL", "database", "slow query"),
    ("PostgreSQL", "database", "connection exhaustion"),
    ("MongoDB", "database", "slow query"),
    ("MongoDB", "database", "connection failure"),
    ("Redis", "cache", "cache miss spike"),
    ("Redis", "cache", "connection failure"),
    ("Docker", "container", "restart loop"),
    ("Docker", "container", "out-of-memory termination"),
    ("Linux", "server", "port already in use"),
    ("Linux", "server", "disk space exhaustion"),
    ("Nginx", "reverse proxy", "502 response"),
    ("Nginx", "reverse proxy", "TLS configuration failure"),
    ("React", "frontend", "unnecessary rerenders"),
    ("React", "frontend", "API loading failure"),
    ("Next.js", "application", "server-side error"),
    ("Next.js", "application", "slow page response"),
    ("Supabase", "database", "authentication failure"),
    ("Supabase", "database", "row-level security issue"),
    ("Git", "repository", "incorrect branch history"),
    ("Git", "repository", "merge conflict"),
    ("CI/CD", "pipeline", "failed deployment"),
    ("CI/CD", "pipeline", "failing test stage"),
    ("AWS", "infrastructure", "service timeout"),
    ("AWS", "infrastructure", "unexpected resource usage"),
]


VENTURE_SCENARIOS = [
    ("B2B SaaS", "market_analysis"),
    ("B2B SaaS", "traction"),
    ("B2B SaaS", "unit_economics"),
    ("B2B SaaS", "retention"),
    ("Marketplace", "market_analysis"),
    ("Marketplace", "growth"),
    ("Marketplace", "unit_economics"),
    ("Marketplace", "risk_analysis"),
    ("Fintech", "market_analysis"),
    ("Fintech", "risk_analysis"),
    ("Fintech", "traction"),
    ("Fintech", "fundraising"),
    ("Healthtech", "market_analysis"),
    ("Healthtech", "risk_analysis"),
    ("Healthtech", "retention"),
    ("Insurtech", "market_analysis"),
    ("Insurtech", "unit_economics"),
    ("Insurtech", "risk_analysis"),
    ("Consumer", "growth"),
    ("Consumer", "retention"),
    ("Consumer", "cohort"),
    ("D2C", "unit_economics"),
    ("D2C", "growth"),
    ("D2C", "retention"),
    ("AI startup", "market_analysis"),
    ("AI startup", "traction"),
    ("AI startup", "unit_economics"),
    ("AI startup", "defensibility"),
    ("Developer tools", "market_analysis"),
    ("Developer tools", "traction"),
    ("Developer tools", "retention"),
    ("Logistics", "market_analysis"),
    ("Logistics", "unit_economics"),
    ("Logistics", "risk_analysis"),
    ("Climate tech", "market_analysis"),
    ("Climate tech", "fundraising"),
    ("Climate tech", "risk_analysis"),
]


def make_developer_record(
    index: int,
    technology: str,
    component: str,
    failure: str,
) -> dict:

    task = "debugging"

    instruction = (
        f"Diagnose a {failure} problem in a {technology} "
        f"{component}."
    )

    context = (
        f"A production system using {technology} has a reported "
        f"{failure} problem involving its {component}. "
        "The developer needs to identify the root cause and "
        "verify the fix."
    )

    output = (
        "Reproduce the failure, inspect relevant logs and metrics, "
        "identify the failing component, isolate the root cause, "
        "apply the smallest appropriate fix, and verify the result "
        "with a regression test or reproducible operational check. "
        "Do not claim the issue is resolved without evidence."
    )

    return {
        "sample_id": f"v02_expand_dev_{index:04d}",
        "domain": "developer",
        "task": task,
        "language": "en",
        "input": instruction,
        "context": context,
        "expected_output": output,
        "metadata": {
            "difficulty": "medium",
            "synthetic": True,
            "source_type": "project_generated",
            "technology": technology,
            "component": component,
        },
        "provenance": {
            "source": "VITO project-generated bootstrap corpus",
            "license": "project_generated",
        },
        "verification": {
            "required": True,
            "method": "human_review_and_task_specific_check",
        },
    }


def make_venture_record(
    index: int,
    company_type: str,
    task: str,
) -> dict:

    instruction = (
        f"Evaluate {task.replace('_', ' ')} for a "
        f"{company_type} startup."
    )

    context = (
        f"The company operates as a {company_type} startup. "
        "The available information is incomplete and should be "
        "distinguished from independently verified evidence."
    )

    outputs = {
        "market_analysis": (
            "Define the relevant customer segment, distinguish "
            "TAM from the realistically serviceable market, "
            "identify alternatives and competitors, and record "
            "the evidence supporting each material conclusion."
        ),
        "traction": (
            "Inspect the reported customer, revenue, usage, and "
            "growth metrics. Separate reported claims from "
            "independently verified measurements and examine the "
            "period over which growth occurred."
        ),
        "unit_economics": (
            "Calculate relevant acquisition, contribution, "
            "retention, and lifetime-value metrics from the "
            "available inputs. State assumptions and avoid "
            "inventing unavailable values."
        ),
        "retention": (
            "Define the cohort and retention period, calculate "
            "retention from the available observations, and "
            "investigate churn, reactivation, and cohort "
            "differences."
        ),
        "growth": (
            "Calculate growth using clearly defined periods and "
            "metrics. Distinguish sequential growth from compound "
            "growth and state the calculation used."
        ),
        "risk_analysis": (
            "Identify material operational, market, financial, "
            "regulatory, concentration, and execution risks. "
            "Quantify exposure when possible and identify the "
            "evidence required to reduce uncertainty."
        ),
        "fundraising": (
            "Record the financing claim, date, instrument, "
            "valuation information, and source. Distinguish "
            "company-reported fundraising information from "
            "independently verified evidence."
        ),
        "defensibility": (
            "Examine technical differentiation, proprietary data, "
            "distribution advantages, switching costs, network "
            "effects, and execution evidence. Do not treat a "
            "marketing claim as proof of defensibility."
        ),
        "cohort": (
            "Group customers by acquisition period and compare "
            "retention, revenue, or engagement across equivalent "
            "time intervals. Clearly document cohort definitions."
        ),
    }

    output = outputs.get(
        task,
        (
            "Identify the relevant metrics, define the analysis "
            "period, separate verified facts from claims, calculate "
            "only metrics supported by available data, and document "
            "missing information."
        ),
    )

    return {
        "sample_id": f"v02_expand_venture_{index:04d}",
        "domain": "venture",
        "task": task,
        "language": "en",
        "input": instruction,
        "context": context,
        "expected_output": output,
        "metadata": {
            "difficulty": "medium",
            "synthetic": True,
            "source_type": "project_generated",
            "company_type": company_type,
        },
        "provenance": {
            "source": "VITO project-generated bootstrap corpus",
            "license": "project_generated",
        },
        "verification": {
            "required": True,
            "method": "human_review_and_task_specific_check",
        },
    }


def main() -> None:

    existing = []

    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                existing.append(json.loads(line))

    records = list(existing)

    index = 1

    for technology, component, failure in DEVELOPER_SCENARIOS:
        records.append(
            make_developer_record(
                index,
                technology,
                component,
                failure,
            )
        )
        index += 1

    index = 1

    for company_type, task in VENTURE_SCENARIOS:
        records.append(
            make_venture_record(
                index,
                company_type,
                task,
            )
        )
        index += 1

    # Prevent accidental duplicate IDs.
    ids = [record["sample_id"] for record in records]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id detected.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print(f"Existing records: {len(existing)}")
    print(f"Expanded records: {len(records)}")
    print(f"Added records: {len(records) - len(existing)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
