#!/usr/bin/env python3

"""
Expand the VITO v0.2 bootstrap corpus with deterministic,
domain-specific synthetic scenarios.

These records are explicitly marked synthetic and project-generated.

The generator intentionally creates scenario-specific expected outputs
rather than reusing one answer template for an entire task category.
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


DEVELOPER_OUTPUTS = {
    ("Node.js", "API", "HTTP 500"): (
        "Reproduce the request against the Node.js API, inspect the route "
        "handler and middleware logs, identify the thrown exception and "
        "its input conditions, then fix the failing code path. Verify the "
        "HTTP response with a regression test and confirm that unrelated "
        "API routes still behave correctly."
    ),
    ("Node.js", "database", "connection timeout"): (
        "Inspect Node.js database connection timing, pool configuration, "
        "database availability, and network errors. Reproduce the timeout, "
        "determine whether the problem is pool exhaustion, connectivity, "
        "or query latency, then apply the smallest appropriate fix and "
        "verify successful connections under representative load."
    ),
    ("Node.js", "memory", "memory growth"): (
        "Measure Node.js heap usage over time, reproduce the growth under "
        "a controlled workload, and inspect retained objects, listeners, "
        "timers, and caches. Identify the allocation that remains reachable, "
        "fix the retention path, and verify that heap usage stabilizes after "
        "repeated workload cycles."
    ),
    ("Python", "API", "HTTP 500"): (
        "Reproduce the failing Python API request, inspect the traceback "
        "and request inputs, isolate the exception-producing code path, "
        "and fix the underlying defect rather than masking the error. Add "
        "a regression test for the failing request and verify the expected "
        "HTTP status and response body."
    ),
    ("Python", "database", "connection timeout"): (
        "Inspect Python database client settings, connection pooling, "
        "network reachability, and database logs. Reproduce the timeout "
        "with the same connection path, determine whether connections are "
        "blocked, exhausted, or unreachable, then correct the configuration "
        "or failing dependency and verify recovery with repeated requests."
    ),
    ("Python", "worker", "high CPU usage"): (
        "Profile the Python worker while the CPU spike occurs, identify the "
        "hot function or loop, and compare behavior with the expected workload. "
        "Check for accidental polling, unbounded loops, inefficient processing, "
        "or excessive retries. Apply a targeted fix and verify CPU usage under "
        "the same workload."
    ),
    ("PostgreSQL", "database", "slow query"): (
        "Capture the slow PostgreSQL query and run EXPLAIN ANALYZE against "
        "representative data. Inspect scans, joins, row estimates, sorting, "
        "locking, and index usage. Optimize the query or supporting index, "
        "then rerun the query plan and latency measurement to verify the "
        "improvement."
    ),
    ("PostgreSQL", "database", "connection exhaustion"): (
        "Inspect PostgreSQL active connections, connection states, pool "
        "configuration, and application connection lifecycle. Determine "
        "whether connections are leaked, pooled incorrectly, or simply "
        "undersized for the workload. Correct the lifecycle or pool settings "
        "and verify connection counts remain stable under repeated requests."
    ),
    ("MongoDB", "database", "slow query"): (
        "Capture the MongoDB query and inspect its execution statistics, "
        "filter fields, sort requirements, and available indexes. Reproduce "
        "the latency on representative data, add or adjust an appropriate "
        "index or query shape, and verify the execution time and examined "
        "documents after the change."
    ),
    ("MongoDB", "database", "connection failure"): (
        "Inspect MongoDB client errors, connection configuration, server "
        "availability, authentication, DNS, and network reachability. "
        "Reproduce the failure from the application environment, identify "
        "the failing connection layer, correct it, and verify successful "
        "connections with a controlled health check."
    ),
    ("Redis", "cache", "cache miss spike"): (
        "Compare Redis hit and miss rates with the period before the spike. "
        "Inspect cache keys, TTL behavior, invalidation logic, eviction "
        "metrics, and recent application changes. Reproduce representative "
        "requests, identify why expected keys are absent, correct the cache "
        "behavior, and verify the hit rate returns to the expected range."
    ),
    ("Redis", "cache", "connection failure"): (
        "Inspect Redis client errors, endpoint configuration, authentication, "
        "network reachability, connection pooling, and Redis server health. "
        "Reproduce the failure from the affected runtime, isolate the failing "
        "connection layer, apply the appropriate configuration or service fix, "
        "and verify successful read/write operations."
    ),
    ("Docker", "container", "restart loop"): (
        "Inspect the container exit code, startup logs, health-check result, "
        "environment variables, mounted files, and entrypoint command. Run "
        "the container interactively to reproduce the failure, identify the "
        "startup condition causing the exit, fix it, and verify the container "
        "remains healthy across multiple restarts."
    ),
    ("Docker", "container", "out-of-memory termination"): (
        "Inspect container memory limits, runtime memory metrics, application "
        "heap usage, and the host's OOM events. Reproduce the workload and "
        "determine whether the application leaks memory or the configured "
        "limit is insufficient. Fix the application or resource configuration "
        "and verify stable memory usage under the same workload."
    ),
    ("Linux", "server", "port already in use"): (
        "Identify the process listening on the conflicting Linux port with "
        "the appropriate socket inspection tools. Determine whether the "
        "process is expected, stale, or incorrectly configured. Resolve the "
        "conflict by correcting the service configuration or process lifecycle, "
        "then restart the intended service and verify the expected listener."
    ),
    ("Linux", "server", "disk space exhaustion"): (
        "Inspect filesystem utilization and identify which mount and directories "
        "consume the available space. Distinguish large files from excessive "
        "inode usage, inspect logs and temporary data, and remove or rotate "
        "only data that is safe to delete. Verify recovered capacity and add "
        "monitoring or retention controls to prevent recurrence."
    ),
    ("Nginx", "reverse proxy", "502 response"): (
        "Inspect Nginx error logs and the upstream application's health and "
        "listening address. Verify DNS, port connectivity, upstream protocol, "
        "timeouts, and recent configuration changes. Reproduce the request, "
        "correct the failing upstream or proxy configuration, and verify the "
        "request succeeds through Nginx."
    ),
    ("Nginx", "reverse proxy", "TLS configuration failure"): (
        "Inspect the Nginx TLS configuration, certificate paths, certificate "
        "validity, key matching, supported protocols, and configuration-test "
        "output. Reproduce the TLS handshake failure, correct the invalid "
        "certificate or configuration, run the Nginx configuration test, "
        "reload safely, and verify the endpoint with a TLS client."
    ),
    ("React", "frontend", "unnecessary rerenders"): (
        "Reproduce the rendering issue with React profiling tools and identify "
        "which component and state or prop changes trigger unnecessary renders. "
        "Inspect unstable object references, callbacks, context updates, and "
        "component boundaries. Apply the smallest optimization that preserves "
        "correct behavior and verify render counts before and after the change."
    ),
    ("React", "frontend", "API loading failure"): (
        "Inspect the browser network request, request URL, HTTP status, "
        "authentication state, response body, and frontend error handling. "
        "Reproduce the request outside the component when possible, determine "
        "whether the failure is client-side or API-side, fix the failing "
        "layer, and verify loading, success, and error states."
    ),
    ("Next.js", "application", "server-side error"): (
        "Inspect the Next.js server logs and reproduce the affected route "
        "with the same server-side inputs. Trace the exception through the "
        "server component, route handler, data access, or middleware layer. "
        "Fix the underlying failure and verify both the server response and "
        "the relevant regression case."
    ),
    ("Next.js", "application", "slow page response"): (
        "Measure the Next.js request and identify whether latency comes from "
        "server rendering, database access, external APIs, middleware, or "
        "unnecessary work. Profile the slow path, optimize the actual bottleneck, "
        "and compare request latency before and after the change under a "
        "representative workload."
    ),
    ("Supabase", "database", "authentication failure"): (
        "Inspect the Supabase authentication response, client configuration, "
        "session state, redirect settings, and relevant auth logs. Reproduce "
        "the exact authentication flow, identify whether the failure occurs "
        "during credentials, token handling, or session persistence, then fix "
        "the affected layer and verify a complete login/logout cycle."
    ),
    ("Supabase", "database", "row-level security issue"): (
        "Reproduce the database request using the affected user's authenticated "
        "session and inspect the applicable Row Level Security policies. Verify "
        "the table relationships and policy predicates, determine why the "
        "request is incorrectly allowed or denied, correct the policy, and "
        "test both authorized and unauthorized access paths."
    ),
    ("Git", "repository", "incorrect branch history"): (
        "Inspect the branch graph, recent commits, remotes, and intended branch "
        "base before changing history. Identify whether the issue came from an "
        "incorrect merge, rebase, or branch point. Choose the least destructive "
        "history correction appropriate to the repository workflow and verify "
        "the resulting commit graph and working tree."
    ),
    ("Git", "repository", "merge conflict"): (
        "Inspect the conflicting files and understand the intended changes from "
        "both branches before resolving them. Resolve each conflict according "
        "to the desired behavior rather than choosing one side blindly, run the "
        "relevant tests, and verify the final diff and commit state."
    ),
    ("CI/CD", "pipeline", "failed deployment"): (
        "Inspect the failed CI/CD job logs and identify the first meaningful "
        "failure rather than the final cascading error. Compare the deployment "
        "environment, secrets, artifact, dependency versions, and target "
        "configuration with the successful baseline. Correct the root cause "
        "and verify the deployment in a controlled run."
    ),
    ("CI/CD", "pipeline", "failing test stage"): (
        "Inspect the failing test output and reproduce the same test command "
        "locally or in an equivalent environment. Determine whether the failure "
        "is caused by the code, test assumptions, environment, timing, or "
        "dependency changes. Fix the underlying issue and verify the complete "
        "test stage passes consistently."
    ),
    ("AWS", "infrastructure", "service timeout"): (
        "Trace the AWS request path and inspect service logs, metrics, network "
        "configuration, timeouts, dependency health, and recent infrastructure "
        "changes. Reproduce the timeout, isolate the slow or unreachable "
        "component, correct the underlying configuration or service issue, "
        "and verify successful requests with latency measurements."
    ),
    ("AWS", "infrastructure", "unexpected resource usage"): (
        "Identify which AWS resource is consuming unexpected capacity or cost "
        "using service metrics and billing or usage data. Compare current usage "
        "with the historical baseline, inspect recent deployments and scaling "
        "changes, determine the source of the increase, and apply a controlled "
        "remediation. Verify usage returns toward the expected baseline."
    ),
}


VENTURE_CONTEXTS = {
    "B2B SaaS": (
        "The company sells recurring software subscriptions to business "
        "customers and reports usage, revenue, and customer metrics."
    ),
    "Marketplace": (
        "The company connects two participant groups and may generate revenue "
        "through transaction fees, subscriptions, or other marketplace charges."
    ),
    "Fintech": (
        "The company provides a financial product or infrastructure service "
        "and may face regulatory, fraud, credit, liquidity, or trust constraints."
    ),
    "Healthtech": (
        "The company provides a healthcare-related product where adoption, "
        "clinical workflow, privacy, regulation, and evidence quality can matter."
    ),
    "Insurtech": (
        "The company applies technology to insurance workflows or products "
        "where underwriting, claims, regulation, distribution, and loss ratios "
        "may affect performance."
    ),
    "Consumer": (
        "The company targets consumers and tracks acquisition, engagement, "
        "retention, monetization, and cohort behavior."
    ),
    "D2C": (
        "The company sells directly to consumers and must understand order "
        "economics, repeat purchasing, contribution margin, and acquisition costs."
    ),
    "AI startup": (
        "The company uses AI as a material part of its product and may depend "
        "on model quality, inference economics, proprietary data, or workflow "
        "integration."
    ),
    "Developer tools": (
        "The company provides software for developers and depends on adoption, "
        "technical workflow integration, retention, and developer distribution."
    ),
    "Logistics": (
        "The company coordinates physical movement or delivery and therefore "
        "may depend on utilization, route economics, operational reliability, "
        "and geographic density."
    ),
    "Climate tech": (
        "The company addresses a climate or resource-efficiency problem and "
        "may depend on project economics, regulation, infrastructure, and "
        "long adoption cycles."
    ),
}


def make_base_metadata(difficulty: str = "medium") -> dict:
    return {
        "difficulty": difficulty,
        "synthetic": True,
        "source_type": "project_generated",
    }


def make_provenance() -> dict:
    return {
        "source": "VITO project-generated bootstrap corpus",
        "license": "project_generated",
    }


def make_verification() -> dict:
    return {
        "required": True,
        "method": "human_review_and_task_specific_check",
    }


def make_developer_record(
    index: int,
    technology: str,
    component: str,
    failure: str,
) -> dict:
    key = (technology, component, failure)

    output = DEVELOPER_OUTPUTS.get(key)

    if output is None:
        output = (
            f"Reproduce the {failure} in the {technology} {component}, "
            "inspect logs and runtime evidence, isolate the failing layer, "
            "apply the smallest appropriate fix, and verify the result with "
            "a regression test or reproducible operational check."
        )

    input_text = (
        f"Diagnose a {failure} problem in a {technology} "
        f"{component}."
    )

    context = (
        f"A production system uses {technology} for its {component}. "
        f"The team has reported {failure}. "
        "The available report is incomplete, so the developer must reproduce "
        "the problem, identify the root cause, and verify the fix before "
        "claiming resolution."
    )

    metadata = make_base_metadata()
    metadata.update(
        {
            "technology": technology,
            "component": component,
            "failure": failure,
        }
    )

    return {
        "sample_id": f"v02_expand_dev_{index:04d}",
        "domain": "developer",
        "task": "debugging",
        "language": "en",
        "input": input_text,
        "context": context,
        "expected_output": output,
        "metadata": metadata,
        "provenance": make_provenance(),
        "verification": make_verification(),
    }


VENTURE_OUTPUTS = {
    "market_analysis": {
        "B2B SaaS": (
            "Define the target business segment, estimate the realistically "
            "serviceable market rather than relying only on broad TAM claims, "
            "map incumbent and substitute solutions, and verify the assumptions "
            "with customer, pricing, and market evidence."
        ),
        "Marketplace": (
            "Define both sides of the marketplace and the geographic or "
            "category scope. Estimate the serviceable opportunity, identify "
            "competing channels and substitutes, and test whether sufficient "
            "supply and demand can form a viable market at the proposed take rate."
        ),
        "Fintech": (
            "Define the customer segment and financial workflow being addressed, "
            "size the relevant serviceable market, identify incumbent financial "
            "products and substitutes, and verify market assumptions against "
            "regulatory, customer, and transaction evidence."
        ),
        "Healthtech": (
            "Define the healthcare stakeholder and workflow being targeted, "
            "separate broad healthcare spending from the addressable market, "
            "identify existing clinical or administrative alternatives, and "
            "record evidence for adoption, reimbursement, and regulatory assumptions."
        ),
        "Insurtech": (
            "Define the insurance segment and workflow, distinguish total "
            "insurance volume from the realistically addressable opportunity, "
            "map incumbent insurers, brokers, and software alternatives, and "
            "verify assumptions using distribution, pricing, and regulatory evidence."
        ),
        "AI startup": (
            "Define the specific customer workflow improved by AI, estimate the "
            "serviceable market for that workflow, distinguish AI-native competitors "
            "from incumbent substitutes, and verify whether customers have a "
            "measurable reason to adopt the product."
        ),
        "Developer tools": (
            "Define the developer persona and workflow, estimate the reachable "
            "developer-tool market, identify competing tools and open-source "
            "substitutes, and evaluate evidence for willingness to pay, adoption, "
            "and switching behavior."
        ),
        "Logistics": (
            "Define the logistics segment, geography, and customer type, estimate "
            "the serviceable transaction or contract opportunity, identify incumbent "
            "operators and software alternatives, and verify assumptions about "
            "density, pricing, and operational constraints."
        ),
        "Climate tech": (
            "Define the climate problem, customer, geography, and purchasing "
            "workflow, distinguish broad climate spending from the addressable "
            "market, identify incumbent solutions, and verify demand against "
            "economic, regulatory, and deployment evidence."
        ),
        "Consumer": (
            "Define the target consumer segment and use case, estimate the "
            "realistically reachable market, identify competing products and "
            "substitutes, and validate market assumptions using observed customer "
            "behavior, pricing, and acquisition evidence."
        ),
        "D2C": (
            "Define the product category and target customer, estimate the "
            "addressable demand rather than relying on broad category spending, "
            "map direct and substitute competitors, and verify assumptions using "
            "pricing, repeat purchase, and acquisition evidence."
        ),
    },
    "traction": {
        "B2B SaaS": (
            "Review customer count, ARR or MRR, net revenue growth, expansion, "
            "and customer concentration over clearly defined periods. Separate "
            "company-reported figures from independently verified evidence and "
            "check whether growth reflects new customers, expansion, or both."
        ),
        "Marketplace": (
            "Inspect buyers, sellers, orders, GMV, take rate, and transaction "
            "growth over comparable periods. Distinguish reported marketplace "
            "activity from verified transactions and determine whether both "
            "sides of the marketplace are growing sustainably."
        ),
        "Fintech": (
            "Inspect active users, transaction volume, revenue, balances, or "
            "other relevant operating metrics over defined periods. Separate "
            "reported figures from verified measurements and check whether growth "
            "is driven by new users, increased usage, or changes in transaction mix."
        ),
        "AI startup": (
            "Inspect customer count, recurring revenue, usage, inference volume, "
            "and expansion over comparable periods. Separate reported claims from "
            "verified measurements and determine whether adoption represents "
            "repeat production usage rather than experimentation alone."
        ),
        "Developer tools": (
            "Inspect active developers, installations, usage, paid conversion, "
            "revenue, and retention over defined periods. Distinguish downloads "
            "or signups from meaningful active usage and separate reported claims "
            "from independently verified measurements."
        ),
        "Consumer": (
            "Inspect active users, orders, revenue, engagement, and repeat "
            "behavior over comparable periods. Separate acquisition volume from "
            "retained activity and distinguish reported metrics from verified "
            "measurements."
        ),
    },
    "unit_economics": {
        "B2B SaaS": (
            "Calculate CAC, gross margin, contribution margin, churn, and "
            "customer lifetime value from the available subscription data. "
            "State the period and assumptions for each metric and do not infer "
            "lifetime value when retention evidence is insufficient."
        ),
        "Marketplace": (
            "Calculate contribution per transaction using GMV, take rate, "
            "variable fulfillment or payment costs, and acquisition costs. "
            "Separate marketplace revenue from GMV and state assumptions before "
            "estimating customer or transaction lifetime value."
        ),
        "Fintech": (
            "Calculate unit contribution using relevant transaction revenue, "
            "servicing costs, fraud or credit losses, acquisition cost, and "
            "retention assumptions. State which costs are included and avoid "
            "treating gross transaction volume as revenue."
        ),
        "Healthtech": (
            "Calculate contribution per customer or covered account using "
            "revenue, delivery costs, support costs, acquisition costs, and "
            "retention assumptions. Clearly distinguish gross revenue from "
            "contribution and identify missing cost inputs."
        ),
        "Insurtech": (
            "Calculate unit economics using premium or fee revenue, acquisition "
            "costs, claims or loss costs where applicable, servicing costs, and "
            "retention. State whether the calculation uses gross or net economics "
            "and identify unavailable risk-cost inputs."
        ),
        "AI startup": (
            "Calculate contribution using customer revenue and variable AI "
            "inference, infrastructure, support, and acquisition costs. Measure "
            "usage on the same period as revenue and state assumptions about "
            "retention before estimating lifetime value."
        ),
        "D2C": (
            "Calculate contribution per order using selling price, discounts, "
            "product cost, fulfillment, payment fees, returns, and acquisition "
            "cost. Separate gross margin from contribution margin and state "
            "which variable costs are included."
        ),
        "Logistics": (
            "Calculate contribution per delivery or route using revenue, driver "
            "or carrier cost, fuel, fulfillment, payment, and acquisition costs. "
            "Account for utilization and route density and avoid treating gross "
            "delivery value as contribution."
        ),
    },
    "retention": {
        "B2B SaaS": (
            "Define the customer cohort and retention window, calculate logo "
            "retention and, where data supports it, revenue retention. Investigate "
            "churn, expansion, contraction, and reactivation separately and "
            "compare cohorts over equivalent periods."
        ),
        "Marketplace": (
            "Define buyer and seller cohorts separately and measure repeat "
            "activity over equivalent periods. Distinguish transaction frequency "
            "from user retention and investigate whether changes are caused by "
            "churn, reactivation, or marketplace supply conditions."
        ),
        "Healthtech": (
            "Define the patient, provider, or customer cohort and retention "
            "window appropriate to the product. Calculate observed retention, "
            "separate scheduled repeat use from genuine re-engagement, and "
            "compare cohort behavior without inventing missing observations."
        ),
        "Insurtech": (
            "Define the policyholder or customer cohort and renewal period. "
            "Calculate observed renewal or retention, distinguish churn from "
            "policy expiration or portfolio changes, and compare equivalent "
            "cohorts using the available evidence."
        ),
        "Consumer": (
            "Define the acquisition cohort and retention intervals, calculate "
            "returning-user or repeat-purchase retention, and investigate churn "
            "and reactivation. Compare cohorts using equivalent observation windows."
        ),
        "D2C": (
            "Define the first-purchase cohort and repeat-purchase window, "
            "calculate customer retention or repeat purchase rate, and separate "
            "new-customer acquisition from returning-customer revenue. State "
            "observation limits for recent cohorts."
        ),
        "Developer tools": (
            "Define the developer acquisition cohort and equivalent activity "
            "windows, measure retained active usage rather than installations "
            "alone, and investigate churn, reactivation, and differences between "
            "free and paid users."
        ),
    },
    "growth": {
        "Marketplace": (
            "Calculate growth separately for GMV, orders, buyers, sellers, and "
            "revenue using clearly defined comparable periods. Distinguish "
            "sequential growth from year-over-year growth and determine whether "
            "growth is driven by volume, take rate, or participant expansion."
        ),
        "Consumer": (
            "Calculate growth for users, orders, revenue, and engagement using "
            "explicit comparable periods. Distinguish sequential from year-over-year "
            "growth and identify whether the change comes from acquisition, "
            "retention, frequency, or monetization."
        ),
        "D2C": (
            "Calculate growth in orders, revenue, average order value, and "
            "repeat purchases over defined periods. Separate sequential growth "
            "from compound growth and identify whether changes come from traffic, "
            "conversion, order frequency, or pricing."
        ),
    },
    "risk_analysis": {
        "Marketplace": (
            "Assess liquidity, supply-demand imbalance, fraud, concentration, "
            "operational, financial, and regulatory risks. Quantify exposure "
            "where possible and identify the evidence needed to determine whether "
            "the marketplace can remain healthy under adverse conditions."
        ),
        "Fintech": (
            "Assess regulatory, fraud, credit, liquidity, financial, operational, "
            "and concentration risks. Quantify exposure where possible and "
            "distinguish documented controls from management assertions."
        ),
        "Healthtech": (
            "Assess clinical, privacy, regulatory, adoption, operational, "
            "financial, and concentration risks. Identify which risks require "
            "external evidence, controlled studies, compliance documentation, "
            "or customer evidence."
        ),
        "Insurtech": (
            "Assess underwriting, claims, regulatory, distribution, concentration, "
            "and operational risks. Quantify exposure where possible and distinguish "
            "observed loss performance from assumptions about future risk."
        ),
        "Logistics": (
            "Assess operational reliability, utilization, geographic concentration, "
            "fuel or labor cost exposure, customer concentration, and regulatory "
            "risks. Quantify sensitivity to major cost or volume changes where data "
            "allows."
        ),
        "Climate tech": (
            "Assess deployment, capital intensity, regulatory, technology, project, "
            "customer concentration, and execution risks. Quantify exposure where "
            "possible and identify evidence needed to validate project economics "
            "and adoption."
        ),
    },
    "fundraising": {
        "Fintech": (
            "Record the financing round, date, instrument, amount, valuation "
            "information, participating investors, and source. Separate company "
            "announcements from independently corroborated records and identify "
            "any missing terms."
        ),
        "Climate tech": (
            "Record the financing amount, date, round or instrument, valuation "
            "information if available, investors, and source. Distinguish "
            "announced commitments from completed financing and identify terms "
            "that remain unverified."
        ),
    },
    "defensibility": {
        "AI startup": (
            "Examine model performance, proprietary data, workflow integration, "
            "switching costs, distribution, and infrastructure or operational "
            "advantages. Separate durable evidence from generic AI feature claims "
            "and identify what competitors could reproduce."
        ),
    },
    "cohort": {
        "Consumer": (
            "Group customers by acquisition period and compare retention, revenue, "
            "orders, or engagement across equivalent observation windows. Define "
            "the cohort entry event and distinguish mature cohorts from recently "
            "acquired users."
        ),
    },
}


def make_venture_record(
    index: int,
    company_type: str,
    task: str,
) -> dict:
    context_description = VENTURE_CONTEXTS.get(
        company_type,
        f"The company operates as a {company_type} startup.",
    )

    task_outputs = VENTURE_OUTPUTS.get(task, {})
    output = task_outputs.get(company_type)

    if output is None:
        output = (
            f"Define the relevant metrics for {task.replace('_', ' ')}, "
            "state the analysis period, separate verified facts from company "
            "claims, calculate only metrics supported by the available data, "
            "and explicitly document missing information."
        )

    input_text = (
        f"Evaluate {task.replace('_', ' ')} for a "
        f"{company_type} startup."
    )

    context = (
        f"{context_description} "
        "The available information is incomplete and should be distinguished "
        "from independently verified evidence."
    )

    metadata = make_base_metadata()
    metadata.update(
        {
            "company_type": company_type,
            "analysis_focus": task,
        }
    )

    return {
        "sample_id": f"v02_expand_venture_{index:04d}",
        "domain": "venture",
        "task": task,
        "language": "en",
        "input": input_text,
        "context": context,
        "expected_output": output,
        "metadata": metadata,
        "provenance": make_provenance(),
        "verification": make_verification(),
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