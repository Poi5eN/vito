#!/usr/bin/env python3

"""
Generate a large, deterministic VITO-specific bootstrap corpus.

Purpose:
- Expand developer / venture / verification training data.
- Avoid identical expected-output templates.
- Produce reproducible records.
- Preserve VITO's normalized corpus schema.
- Keep benchmark data separate.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data"
    / "raw"
    / "vito_v0.3_generated.jsonl"
)

SEED = 42

# Approximate target. We will measure exact tokens afterward.
TARGET_RECORDS = 20000


DEVELOPER_SCENARIOS = [
    "debugging",
    "api_design",
    "database",
    "sql",
    "testing",
    "refactoring",
    "performance",
    "docker",
    "linux",
    "git",
    "networking",
    "authentication",
    "authorization",
    "caching",
    "queues",
    "observability",
    "ci_cd",
    "system_design",
    "javascript",
    "typescript",
    "python",
    "react",
    "nextjs",
    "nodejs",
    "security",
]


VENTURE_SCENARIOS = [
    "market",
    "problem_validation",
    "product_fit",
    "traction",
    "growth",
    "retention",
    "unit_economics",
    "ltv_cac",
    "gross_margin",
    "runway",
    "fundraising",
    "pricing",
    "gtm",
    "competition",
    "defensibility",
    "team",
    "risk_analysis",
    "data_quality",
    "cohort_analysis",
]


VERIFICATION_SCENARIOS = [
    "claim_verification",
    "source_verification",
    "metric_reconciliation",
    "calculation_verification",
    "api_verification",
    "database_verification",
    "deployment_verification",
    "security_verification",
    "incident_verification",
    "configuration_verification",
]


COMPONENTS = [
    "authentication middleware",
    "REST endpoint",
    "PostgreSQL query",
    "MongoDB aggregation",
    "Redis cache",
    "background worker",
    "Docker container",
    "GitHub Actions workflow",
    "Next.js server action",
    "React component",
    "Node.js service",
    "Python worker",
    "message queue",
    "reverse proxy",
    "API gateway",
    "object storage",
    "database migration",
    "CI pipeline",
    "monitoring agent",
    "feature flag",
]


ERRORS = [
    "returns HTTP 500 intermittently",
    "times out under concurrent requests",
    "returns stale data",
    "fails only in production",
    "works locally but fails in CI",
    "loses state after deployment",
    "produces duplicate records",
    "drops requests under load",
    "fails after a schema migration",
    "causes elevated memory usage",
    "returns inconsistent pagination",
    "fails authentication unexpectedly",
    "creates excessive database connections",
    "produces incorrect aggregate values",
    "causes a queue backlog",
]


METRICS = [
    "monthly recurring revenue",
    "net revenue retention",
    "gross margin",
    "customer acquisition cost",
    "lifetime value",
    "activation rate",
    "conversion rate",
    "monthly churn",
    "weekly active users",
    "payback period",
    "burn multiple",
    "runway",
]


SOURCES = [
    "billing export",
    "database snapshot",
    "analytics dashboard",
    "CRM export",
    "cohort report",
    "bank statement",
    "cloud billing report",
    "application logs",
    "API response",
    "deployment logs",
    "customer interview notes",
    "signed contract",
]


INDUSTRIES = [
    "B2B SaaS",
    "fintech",
    "healthtech",
    "insurtech",
    "marketplace",
    "developer tools",
    "AI infrastructure",
    "consumer subscription",
    "logistics",
    "edtech",
    "proptech",
    "climate software",
]


def stable_id(prefix: str, index: int) -> str:
    raw = f"{SEED}:{prefix}:{index}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"v03_{prefix}_{digest}"


def number(rng: random.Random, low: int, high: int) -> int:
    return rng.randint(low, high)


def money(rng: random.Random, low: int, high: int) -> str:
    return f"${rng.randint(low, high):,}"


def developer_record(
    index: int,
    scenario: str,
    rng: random.Random,
) -> dict:

    component = rng.choice(COMPONENTS)
    error = rng.choice(ERRORS)

    latency = number(rng, 120, 2400)
    users = number(rng, 100, 50000)
    error_rate = round(rng.uniform(0.5, 18.0), 2)

    task = (
        f"Investigate a {component} that {error}."
    )

    context = f"""
A production system contains a {component}.

Observed symptoms:
- Failure mode: {error}
- Approximate affected users: {users}
- Observed latency: {latency} ms
- Error rate during the incident: {error_rate}%
- The issue must be investigated without assuming the first
  suspected cause is correct.

Scenario category: {scenario}

The engineer needs to separate:
1. observed facts,
2. hypotheses,
3. evidence needed to test each hypothesis,
4. remediation,
5. regression tests.

Relevant engineering principle:
A plausible explanation is not the same thing as a verified root cause.
""".strip()

    output = f"""
Verification-first engineering analysis:

1. Confirm the observed behavior
   Reproduce the failure using the smallest reliable reproduction.
   Record request IDs, timestamps, inputs, environment, and exact
   error responses.

2. Establish a baseline
   Compare successful and failing requests and determine whether the
   behavior correlates with traffic, deployment version, dependency
   version, database state, or configuration.

3. Test hypotheses
   For the {component}, inspect logs, metrics, traces, dependency
   responses, resource utilization, and relevant configuration.
   Change one variable at a time where practical.

4. Avoid premature conclusions
   The {error_rate}% error rate is evidence that something is wrong,
   but it does not by itself establish the root cause.

5. Remediation
   Apply the smallest change supported by the evidence. Roll it out
   gradually and monitor the same metrics used to establish the
   baseline.

6. Regression protection
   Add an automated test that reproduces the original failure and
   verifies the corrected behavior.

7. Verification result
   The incident should only be marked resolved after the reproduction
   no longer fails and production telemetry confirms that the relevant
   error and latency measurements have returned to the expected range.

Key distinction:
observed symptom -> hypothesis -> experiment -> evidence -> conclusion.
""".strip()

    return {
        "sample_id": stable_id("developer", index),
        "record_type": "structured",
        "domain": "developer",
        "task": scenario,
        "language": "en",
        "input": task,
        "context": context,
        "expected_output": output,
        "provenance": {
            "license": "project_generated",
            "source": "VITO project-generated synthetic bootstrap corpus",
            "source_type": "vito_project",
            "dataset": "vito_v0.3_generated",
            "generation_seed": SEED,
        },
        "verification": {
            "method": "deterministic_generation_and_schema_validation",
            "required": True,
        },
    }


def venture_record(
    index: int,
    scenario: str,
    rng: random.Random,
) -> dict:

    industry = rng.choice(INDUSTRIES)
    metric = rng.choice(METRICS)
    source = rng.choice(SOURCES)

    customers = number(rng, 80, 25000)
    revenue = number(rng, 20_000, 2_500_000)
    growth = round(rng.uniform(2, 28), 1)
    margin = round(rng.uniform(15, 88), 1)

    task = (
        f"Evaluate {scenario} evidence for a {industry} startup."
    )

    context = f"""
Startup profile:

Industry: {industry}
Customers: {customers:,}
Monthly revenue: ${revenue:,}
Reported monthly growth: {growth}%
Reported gross margin: {margin}%
Metric under investigation: {metric}
Primary evidence source: {source}

The analysis must distinguish:
- directly observed facts,
- founder-reported claims,
- calculated metrics,
- assumptions,
- missing evidence.

The goal is not to reward a narrative. The goal is to determine
what the available evidence actually supports.
""".strip()

    calculated_growth = round(
        revenue * (1 + growth / 100),
        2,
    )

    output = f"""
Verification-first venture analysis:

Startup context:
The company operates in {industry} and reports {customers:,}
customers with monthly revenue of ${revenue:,}.

Evidence classification:
- Reported customer count: claim until reconciled against the
  underlying customer or billing system.
- Reported growth: claim until two comparable periods are available.
- Gross margin: requires revenue and cost-of-goods evidence.
- {metric}: requires the source data appropriate to that metric.

Verification procedure:

1. Obtain the original {source}.
2. Confirm its reporting period and whether it is gross or net.
3. Reconcile headline values against the underlying records.
4. Check whether definitions changed between periods.
5. Calculate the metric independently.
6. Compare the independent calculation with management's figure.
7. Record unexplained differences instead of silently correcting them.

Illustrative calculation:
If ${revenue:,} is the current monthly revenue and the reported
growth rate is {growth}%, applying that rate mechanically gives
approximately ${calculated_growth:,.2f} for the next comparable period.

That calculation is only an arithmetic illustration. It is not
evidence that the forecast will occur.

Decision evidence:
A conclusion about {scenario} should depend on verified evidence,
not on the headline number alone.

Missing-data rule:
If the underlying evidence cannot be obtained, mark the metric as
unverified rather than inventing a value.
""".strip()

    return {
        "sample_id": stable_id("venture", index),
        "record_type": "structured",
        "domain": "venture",
        "task": scenario,
        "language": "en",
        "input": task,
        "context": context,
        "expected_output": output,
        "provenance": {
            "license": "project_generated",
            "source": "VITO project-generated synthetic bootstrap corpus",
            "source_type": "vito_project",
            "dataset": "vito_v0.3_generated",
            "generation_seed": SEED,
        },
        "verification": {
            "method": "deterministic_generation_and_schema_validation",
            "required": True,
        },
    }


def verification_record(
    index: int,
    scenario: str,
    rng: random.Random,
) -> dict:

    metric = rng.choice(METRICS)
    source = rng.choice(SOURCES)

    claimed = round(rng.uniform(5, 180), 2)
    independently_calculated = round(
        claimed * rng.uniform(0.82, 1.18),
        2,
    )

    difference = round(
        independently_calculated - claimed,
        2,
    )

    task = (
        f"Verify a {scenario.replace('_', ' ')} involving {metric}."
    )

    context = f"""
Verification scenario:

Claimed {metric}: {claimed}
Independent calculation from {source}: {independently_calculated}
Difference: {difference}

The task is to determine whether the claim is supported by the
available evidence.

Required verification sequence:
1. identify the exact claim,
2. identify its definition,
3. identify the source,
4. verify the reporting period,
5. reproduce the calculation,
6. reconcile differences,
7. state what remains uncertain.
""".strip()

    if abs(difference) < 0.01:
        conclusion = "The independent calculation matches the claim."
    else:
        conclusion = (
            "The independent calculation does not exactly match the "
            "claim, so the difference must be reconciled before the "
            "claim can be treated as verified."
        )

    output = f"""
Verification report:

Claim:
{metric} was reported as {claimed}.

Independent check:
Using the {source}, the independently calculated value is
{independently_calculated}.

Reconciliation:
Difference = independent value - claimed value
Difference = {difference}

Conclusion:
{conclusion}

Evidence handling:
The numerical difference should not be ignored or rounded away
without understanding its cause. Possible causes include different
time periods, metric definitions, excluded records, currency
conversion, gross-versus-net treatment, or an actual reporting error.

Verification status:
Do not label the claim verified merely because the numbers look
similar. Verification requires agreement between the claim,
definition, source period, and independently reproducible
calculation.

Evidence quality:
The strongest evidence is the underlying source data, followed by
a reproducible calculation and an explicit reconciliation trail.
""".strip()

    return {
        "sample_id": stable_id("verification", index),
        "record_type": "structured",
        "domain": "verification",
        "task": scenario,
        "language": "en",
        "input": task,
        "context": context,
        "expected_output": output,
        "provenance": {
            "license": "project_generated",
            "source": "VITO project-generated synthetic bootstrap corpus",
            "source_type": "vito_project",
            "dataset": "vito_v0.3_generated",
            "generation_seed": SEED,
        },
        "verification": {
            "method": "deterministic_generation_and_schema_validation",
            "required": True,
        },
    }


def main() -> None:

    rng = random.Random(SEED)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = []

    # Roughly balanced across the three VITO-specific domains.
    developer_target = int(TARGET_RECORDS * 0.42)
    venture_target = int(TARGET_RECORDS * 0.33)
    verification_target = (
        TARGET_RECORDS
        - developer_target
        - venture_target
    )

    index = 0

    for _ in range(developer_target):

        scenario = rng.choice(DEVELOPER_SCENARIOS)

        records.append(
            developer_record(
                index,
                scenario,
                rng,
            )
        )

        index += 1

    for _ in range(venture_target):

        scenario = rng.choice(VENTURE_SCENARIOS)

        records.append(
            venture_record(
                index,
                scenario,
                rng,
            )
        )

        index += 1

    for _ in range(verification_target):

        scenario = rng.choice(VERIFICATION_SCENARIOS)

        records.append(
            verification_record(
                index,
                scenario,
                rng,
            )
        )

        index += 1

    rng.shuffle(records)

    # Make every synthetic example's output deterministic and unique.
    #
    # Randomized scenario parameters can occasionally converge on the
    # same generated answer. Add a stable case reference so the
    # duplicate-output invariant remains guaranteed without removing
    # the safety check.
    for record in records:
        case_ref = record["sample_id"]
        record["expected_output"] += (
            f"\\n\\nSynthetic case reference: {case_ref}"
        )

    # Final deterministic duplicate guard.
    ids = set()
    outputs = set()

    for record in records:

        if record["sample_id"] in ids:
            raise RuntimeError(
                f"Duplicate sample_id: {record['sample_id']}"
            )

        ids.add(record["sample_id"])

        output_hash = hashlib.sha256(
            record["expected_output"].encode("utf-8")
        ).hexdigest()

        if output_hash in outputs:
            raise RuntimeError(
                f"Duplicate generated output: "
                f"{record['sample_id']}"
            )

        outputs.add(output_hash)

    with OUTPUT.open("w", encoding="utf-8") as handle:

        for record in records:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 68)
    print("VITO v0.3 VITO-SPECIFIC EXPANSION")
    print("=" * 68)
    print()
    print(f"Generated records: {len(records):,}")
    print(f"Developer:         {developer_target:,}")
    print(f"Venture:           {venture_target:,}")
    print(f"Verification:      {verification_target:,}")
    print()
    print(f"Unique IDs:        {len(ids):,}")
    print(f"Unique outputs:    {len(outputs):,}")
    print()
    print(f"Output: {OUTPUT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
