# VITO Startup Score Specification v0.1

**Score:** 0–1000  
**Name:** VITO Startup Score  
**Release:** 0.1.0  
**Creator:** Gourav Kumar Upadhyay

## 1. Purpose

VITO Startup Score is a repeatable analytical score for comparing the evidence available about a startup at a specific point in time. It is an analytical output, not a substitute for investment judgment or professional due diligence.

Every startup receives a score from 0 to 1000 plus an evidence-confidence score and an evidence-coverage report.

## 2. Dimensions and weights

| Dimension | Weight |
|---|---:|
| Market opportunity & urgency | 120 |
| Product & problem-solution fit evidence | 100 |
| Traction | 150 |
| Growth & momentum | 120 |
| Retention & engagement | 100 |
| Unit economics & financial quality | 120 |
| Go-to-market efficiency | 70 |
| Technology & defensibility | 70 |
| Team / execution evidence | 80 |
| Risk, resilience & data quality | 70 |
| **Total** | **1000** |

Each dimension is scored from 0–100 and multiplied by its weight/100.

## 3. Missing-data policy

A score must always be emitted, but missing evidence must be visible. For v0.1, a missing dimension receives a neutral provisional component score of 50 rather than zero. The resulting overall score is marked `provisional=true`, and the confidence score falls as evidence coverage decreases.

This prevents a startup from being mechanically punished just because a particular metric was not supplied, while also preventing the overall number from being presented as equally reliable for every company.

## 4. Evidence confidence

Confidence is a separate 0–100 measure based on:

- evidence coverage across weighted dimensions
- source quality
- freshness of volatile data
- internal consistency
- proportion of calculated values supported by raw inputs
- unresolved contradictions

A high startup score with low confidence must be displayed as a provisional result.

## 5. Source hierarchy

When sources disagree, VITO should prefer, where applicable:

1. Primary company/transaction documents supplied for analysis
2. Government or official registries
3. Audited or regulated filings
4. Direct product/usage evidence
5. Reputable secondary reporting
6. Company social posts and marketing pages
7. Unverified third-party claims

The system must retain the conflicting claims rather than silently discarding them.

## 6. Internet freshness

Every external observation must store:

- `source_url`
- `publisher`
- `published_at` if known
- `retrieved_at`
- `as_of`
- `source_type`
- `reliability_tier`
- `content_hash` when practical

Volatile metrics such as funding, pricing, headcount, web traffic, product availability and market announcements must be timestamped. Historical data must not be silently treated as current.

## 7. Calculation policy

Whenever possible, VITO should calculate metrics from raw data with deterministic tools rather than relying on language-model arithmetic.

Examples include:

- MoM/YoY growth
- CAGR
- MRR/ARR
- gross margin
- CAC
- LTV
- LTV:CAC
- payback period
- burn
- runway
- NRR/GRR
- retention curves
- cohort statistics
- funnel conversion

Each calculated metric should retain the inputs and formula used.

## 8. Score record

The canonical output should look like:

```json
{
  "startup_id": "example-ai",
  "score": 742.6,
  "score_scale": 1000,
  "provisional": true,
  "confidence": 71.4,
  "as_of": "2026-09-29T00:00:00Z",
  "dimension_scores": {},
  "evidence_coverage": {},
  "conflicts": [],
  "missing_evidence": [],
  "sources": [],
  "methodology_version": "0.1.0"
}
```

## 9. What VITO must never do

- Treat an unsupported founder claim as verified fact.
- Pretend missing data was observed.
- Invent market numbers or financial metrics.
- Hide conflicting sources.
- Present the score without its as-of date and methodology version.
- Treat the score as a guaranteed prediction of future performance.
