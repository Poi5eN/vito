# VITO Startup Intelligence Architecture v0.1

VITO Startup Intelligence is a layered system. The language model interprets requests and evidence; deterministic tools perform calculations; a provenance layer records where each observation came from; and the scoring engine produces the canonical 0–1000 result.

## Pipeline

```text
User / Startup Files
        |
        v
Ingestion Layer
(PDF, XLSX, CSV, text, URLs)
        |
        v
Evidence Extraction
        |
        +------------------------+
        |                        |
        v                        v
Structured Metrics         Source Registry
        |                        |
        v                        v
Deterministic Analysis <--- Freshness / Conflicts
        |
        v
Dimension Scoring
        |
        v
VITO Startup Score (0–1000)
        |
        v
Confidence + Evidence Coverage + Missing Data
        |
        v
Report / Dashboard / API
```

## Internet research rule

When VITO uses an external research tool, every material observation becomes an evidence record before it can affect the score. Store the URL, publisher, retrieval time, publication time when known, as-of date, source type, reliability tier, and a stable locator or excerpt. The score is generated from the captured evidence snapshot, not from an unrecorded live browsing session.

## Freshness

Each metric has a volatility class. High-volatility metrics (funding, pricing, headcount, product availability, announcements) must carry explicit timestamps. Lower-volatility facts can use longer validity windows. Historical observations remain historical; they must not be relabeled as current.

## Conflict handling

VITO should preserve conflicting observations. The resolver ranks source quality and recency, but the final report must show unresolved conflicts and avoid silently choosing a value when the evidence is genuinely ambiguous.

## Deterministic computation

The model proposes what to calculate and how to interpret it, but arithmetic and statistical calculations should be performed by deterministic code. Every calculated value stores its inputs and formula.

## Scoring

The canonical score is the weighted sum of ten 0–100 dimension scores defined in `VITO_STARTUP_SCORE_SPEC_V0.1.md`. The score always has an `as_of` timestamp, methodology version, evidence coverage and confidence value.

## Modes

### Analyst mode
Human supplies evidence and/or component scores; VITO computes the result and report.

### Research mode
VITO gathers current external evidence, records it, extracts metrics, resolves conflicts, calculates derived values and then scores.

### Continuous-monitoring mode
A future service can rerun a startup analysis on a schedule and emit a new version only when relevant evidence changes.

## Important separation

The score is an analytical artifact. It is not a guarantee of future startup performance, nor should the model invent missing facts to make a score look complete.
