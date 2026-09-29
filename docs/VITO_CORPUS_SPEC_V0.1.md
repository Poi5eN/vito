# VITO Corpus Specification v0.1

**Project:** VITO — Verification-First Intelligence for Technical Operations  
**Creator:** Gourav Kumar Upadhyay  
**Release:** 0.1.0  
**Status:** Draft / Development

## 1. Purpose

VITO is a developer-focused language-model project with a second capability domain for startup analysis. The corpus is designed around tasks that require understanding, analysis, calculation, verification, and evidence handling instead of generic conversational completion.

## 2. Capability families

### Developer
- Code generation and completion
- Debugging
- Code review
- Refactoring
- Test generation
- SQL and data work
- Linux, Git, Docker, networking and DevOps
- System design and technical explanation
- Verification of proposed fixes

### Venture / startup analysis
- Company and product understanding
- Market and competitor analysis
- Traction and growth analysis
- Retention and cohort analysis
- Unit economics and financial analysis
- GTM analysis
- Technology and defensibility assessment
- Risk and data-quality analysis
- Evidence-backed startup reports
- Detection of contradictions and missing information

## 3. Shared VITO protocol

Training examples should encourage a structured workflow:

1. Understand the task.
2. Identify relevant evidence and assumptions.
3. Form testable hypotheses.
4. Calculate or execute when tools are available.
5. Compare claims against evidence.
6. State uncertainty and missing data.
7. Produce an actionable conclusion.
8. Define how the conclusion can be verified.

This protocol is not intended to expose private chain-of-thought. It trains useful, auditable artifacts such as assumptions, evidence, calculations, checks, actions and verification steps.

## 4. Task record

A training record should use a stable schema such as:

```json
{
  "sample_id": "vito_000001",
  "version": "0.1",
  "domain": "developer|venture",
  "task": "debugging|code_generation|startup_analysis|...",
  "language": "en|hinglish|other",
  "input": "...",
  "context": "...",
  "expected_output": "...",
  "verification": ["..."],
  "evidence": [
    {
      "source_id": "src_001",
      "claim": "..."
    }
  ],
  "provenance": {
    "source_type": "original|licensed|synthetic|public_record",
    "license": "...",
    "creator": "..."
  }
}
```

## 5. Data quality rules

- Keep provenance for every externally derived record.
- Record the license and source URL where applicable.
- Do not put secrets, credentials, private customer data, or unnecessary personal information into the corpus.
- Keep benchmark items isolated from training data.
- Deduplicate before train/validation/test splitting.
- Maintain deterministic dataset versions and manifests.
- Preserve raw source metadata separately from transformed training text.

## 6. Evaluation split

The first release will maintain:

- `train`: language/model learning material
- `validation`: held-out development data
- `test`: frozen, never used for training decisions
- `benchmark`: separate evaluation suites, including VITO-DEV-Bench and VITO-VENTURE-Bench

## 7. Venture-data principle

Startup analysis examples should teach VITO to separate:

- founder/company claim
- observed source fact
- calculated metric
- model inference
- unresolved conflict
- missing evidence

A startup score must never be generated solely from a prose impression when structured evidence is available.
