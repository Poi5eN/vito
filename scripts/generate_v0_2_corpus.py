#!/usr/bin/env python3

"""
Generate a deterministic synthetic development corpus for VITO v0.2.

This is intentionally a bootstrap corpus.

It is NOT intended to replace licensed/public training datasets.
Its purpose is to give VITO a substantially larger, structured,
domain-specific corpus while the external-data pipeline is built.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "data" / "raw" / "vito_v0.2_records.jsonl"


DEVELOPER_TASKS = [
    (
        "debugging",
        "Fix a JavaScript API request that returns HTTP 500.",
        "The frontend calls /api/users and receives HTTP 500. The backend log shows an exception.",
        "Check the backend exception first, reproduce the request locally, inspect the failing code path, fix the underlying exception, and add a regression test for the request.",
    ),
    (
        "debugging",
        "Diagnose a database connection timeout.",
        "An application connects successfully during development but production requests intermittently time out when connecting to PostgreSQL.",
        "Check database reachability, connection limits, network rules, connection pool configuration, and timeout settings. Reproduce the failure and verify the fix with repeated connections.",
    ),
    (
        "debugging",
        "Investigate a Node.js process using excessive memory.",
        "A Node.js service starts normally but memory usage increases continuously over several hours.",
        "Capture heap information, identify retained objects, inspect listeners and caches, reproduce the growth, and verify that memory usage stabilizes after the fix.",
    ),
    (
        "code_review",
        "Review an asynchronous JavaScript function.",
        "async function loadUser(id) { const response = await fetch(`/users/${id}`); return response.json(); }",
        "Verify the HTTP status before parsing the response, handle network failures, validate the returned data when required, and make error handling explicit.",
    ),
    (
        "code_review",
        "Review an Express route that accepts a user identifier.",
        "app.get('/users/:id', async (req, res) => { const user = await getUser(req.params.id); res.json(user); });",
        "Validate the identifier, handle the not-found case, catch asynchronous failures, and return appropriate HTTP status codes.",
    ),
    (
        "code_generation",
        "Write a TypeScript function that groups objects by a property.",
        "The function receives an array of objects and a property name.",
        "Implement a generic function using a typed accumulator and return an object whose keys contain the corresponding objects.",
    ),
    (
        "code_generation",
        "Create a function that removes duplicate strings.",
        "The input is an array of strings and the original order should be preserved.",
        "Use a Set to track values already encountered and return the first occurrence of each string.",
    ),
    (
        "code_generation",
        "Create an Express health-check endpoint.",
        "The service should expose GET /health.",
        "Return HTTP 200 with a small JSON response indicating that the service is running. Keep the endpoint independent of business logic.",
    ),
    (
        "test_generation",
        "Write tests for a function that calculates a percentage.",
        "The function receives a numerator and denominator.",
        "Test normal values, zero numerator, zero denominator, decimal values, and invalid inputs according to the chosen API contract.",
    ),
    (
        "test_generation",
        "Write tests for an authentication endpoint.",
        "POST /login accepts an email and password.",
        "Test successful authentication, invalid credentials, missing fields, malformed email input, rate limiting behavior, and unexpected server errors.",
    ),
    (
        "sql",
        "Find customers who placed more than five orders.",
        "The database contains customers and orders tables linked by customer_id.",
        "Group orders by customer_id, count the orders, filter groups above five, and join the result with customer information.",
    ),
    (
        "sql",
        "Calculate monthly revenue.",
        "An orders table contains order_date and total_amount.",
        "Group completed orders by year and month and calculate SUM(total_amount), while excluding cancelled orders.",
    ),
    (
        "sql",
        "Find the most recent order for every customer.",
        "The orders table contains customer_id and created_at.",
        "Use a window function or a grouped maximum to select the latest order for each customer.",
    ),
    (
        "devops",
        "Diagnose a Docker container that repeatedly restarts.",
        "The container exits shortly after startup and Docker reports a restart loop.",
        "Inspect container logs and exit status, verify environment variables and mounted files, reproduce the startup command manually, and fix the underlying failure.",
    ),
    (
        "devops",
        "Deploy a Node.js service behind a reverse proxy.",
        "The application listens on port 3000 and should be accessible through HTTPS.",
        "Run the application behind a reverse proxy, configure TLS, forward the required headers, expose only the intended public ports, and verify HTTPS and health checks.",
    ),
    (
        "systems",
        "Explain why a process can become CPU bound.",
        "A service uses one CPU core continuously while request latency increases.",
        "Profile the process, identify the hot code path, determine whether the work is computational or caused by an unintended loop, optimize the bottleneck, and benchmark the change.",
    ),
    (
        "systems",
        "Explain connection pooling.",
        "An application creates a new database connection for every request.",
        "A connection pool reuses established connections, reducing connection setup overhead and controlling the number of concurrent database connections.",
    ),
    (
        "refactoring",
        "Refactor duplicated validation logic.",
        "Several API routes independently validate email, phone, and required fields.",
        "Extract reusable validation functions or schemas, keep route-specific rules explicit, and add tests covering shared validation behavior.",
    ),
    (
        "git",
        "Recover a local branch after an accidental commit.",
        "A developer committed changes locally but wants to move the commit to another branch.",
        "Inspect the commit history, create or switch to the intended branch, move the commit using the appropriate Git operation, and verify the resulting history before pushing.",
    ),
    (
        "linux",
        "Find which process is listening on a port.",
        "A service cannot start because port 8000 is already in use.",
        "Use a platform-appropriate socket/process inspection command, identify the process, determine whether it should be stopped, and verify that the port becomes available.",
    ),
    (
        "api",
        "Design pagination for a REST API.",
        "An endpoint may return hundreds of thousands of records.",
        "Return a bounded page size, expose pagination metadata or cursors, define stable ordering, validate pagination parameters, and avoid loading the entire dataset into memory.",
    ),
    (
        "security",
        "Prevent SQL injection in an API.",
        "A request parameter is currently concatenated into a SQL query.",
        "Use parameterized queries or a safe query builder, validate inputs where appropriate, avoid constructing SQL from raw user input, and test malicious input cases.",
    ),
    (
        "verification",
        "Verify that an API bug fix actually works.",
        "A developer claims that a production API bug has been fixed.",
        "Reproduce the original failure, create a regression test, run the relevant test suite, verify the response behavior, and record the evidence used to confirm the fix.",
    ),
]


VENTURE_TASKS = [
    (
        "market_analysis",
        "Evaluate a startup entering a large but competitive market.",
        "The startup targets an established market with several funded competitors.",
        "Separate total market size from the realistically serviceable market, identify the specific customer segment, examine existing alternatives, and determine whether the startup has evidence for differentiated demand.",
    ),
    (
        "traction",
        "Evaluate startup growth.",
        "A company reports monthly revenue of 100,000 followed by 120,000 and then 150,000.",
        "Calculate month-over-month growth for each period, inspect the consistency of the trend, and distinguish reported revenue from independently verified revenue.",
    ),
    (
        "unit_economics",
        "Evaluate customer acquisition economics.",
        "A startup spends 300,000 on acquisition and obtains 600 new customers during the period.",
        "CAC is acquisition spend divided by acquired customers. The result should then be compared with contribution margin, retention, and customer lifetime value rather than evaluated in isolation.",
    ),
    (
        "unit_economics",
        "Calculate LTV using contribution margin.",
        "Average monthly revenue per customer is 1,000, contribution margin is 40%, and average customer lifetime is 12 months.",
        "Calculate monthly contribution per customer and multiply it by the expected lifetime. State the assumptions and distinguish the resulting estimate from observed customer lifetime value.",
    ),
    (
        "runway",
        "Calculate startup runway.",
        "A startup has 12 million in cash and burns 1 million per month.",
        "Basic runway is cash divided by monthly burn, giving 12 months under the assumption that burn and available cash remain otherwise unchanged.",
    ),
    (
        "growth",
        "Calculate compound monthly growth.",
        "Revenue grows from 1 million to 1.728 million over three months.",
        "Calculate the monthly compound growth rate and show the formula used rather than relying on a simple average.",
    ),
    (
        "retention",
        "Evaluate customer retention.",
        "A cohort begins with 1,000 customers and has 700 active customers after six months.",
        "The observed six-month retention is 70%. Investigate cohort definition, reactivation, churn timing, and whether active customers represent paying customers.",
    ),
    (
        "cohort",
        "Explain cohort analysis.",
        "A subscription company wants to understand whether newer customer cohorts retain better than older cohorts.",
        "Group customers by acquisition period and compare retention, revenue, or other relevant metrics across consistent time intervals.",
    ),
    (
        "fundraising",
        "Evaluate a startup fundraising claim.",
        "The founders state that the company raised capital at a particular valuation.",
        "Separate the founder's statement from verified financing evidence, identify the financing date and instrument when available, and record uncertainty when independent evidence is unavailable.",
    ),
    (
        "risk_analysis",
        "Identify risks in a startup analysis.",
        "The company depends heavily on one acquisition channel and one large customer.",
        "Flag concentration risk, investigate channel stability and customer dependency, quantify exposure where possible, and identify evidence that would reduce uncertainty.",
    ),
    (
        "evidence_verification",
        "Verify a startup's reported metric.",
        "A company reports 100% year-over-year growth.",
        "Identify the source, reporting period, metric definition, baseline period, and whether the figure is independently supported. Do not convert an unverified founder claim into an established fact.",
    ),
    (
        "data_quality",
        "Handle missing startup financial data.",
        "A startup provides revenue growth but no gross margin or burn information.",
        "Do not invent the missing metrics. Record them as unavailable, lower confidence where appropriate, and identify the additional evidence required for analysis.",
    ),
    (
        "market_analysis",
        "Distinguish TAM from realistic opportunity.",
        "A startup claims a 10 billion dollar market based on an industry report.",
        "Check how the market is defined, which geography and customer segment are included, the time period, and whether the startup's actual product addresses that entire market.",
    ),
    (
        "risk_analysis",
        "Evaluate regulatory risk.",
        "A startup operates in a regulated industry and plans rapid geographic expansion.",
        "Identify applicable regulatory dependencies, affected markets, compliance requirements, and the evidence needed before treating expansion assumptions as reliable.",
    ),
    (
        "verification",
        "Separate facts from inference in startup analysis.",
        "A startup's website states that it has thousands of customers.",
        "Record the statement as a company claim unless independently verified. Do not infer revenue, retention, or market share from the statement without supporting evidence.",
    ),
]


def make_record(
    index: int,
    domain: str,
    task: str,
    instruction: str,
    context: str,
    output: str,
) -> dict:

    return {
        "sample_id": f"v02_{domain}_{index:04d}",
        "domain": domain,
        "task": task,
        "language": "en",
        "input": instruction,
        "context": context,
        "expected_output": output,
        "metadata": {
            "difficulty": "medium",
            "synthetic": True,
            "source_type": "project_generated",
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

    records = []
    index = 1

    for task, instruction, context, output in DEVELOPER_TASKS:
        records.append(
            make_record(
                index,
                "developer",
                task,
                instruction,
                context,
                output,
            )
        )
        index += 1

    for task, instruction, context, output in VENTURE_TASKS:
        records.append(
            make_record(
                index,
                "venture",
                task,
                instruction,
                context,
                output,
            )
        )
        index += 1

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

    print(f"Wrote {len(records)} records")
    print(f"Developer: {len(DEVELOPER_TASKS)}")
    print(f"Venture: {len(VENTURE_TASKS)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
