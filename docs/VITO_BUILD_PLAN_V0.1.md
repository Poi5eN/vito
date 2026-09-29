# VITO Build Plan v0.1

VITO is being built as a developer-first language model plus a verification, venture-analysis, analytics and serving layer.

## Current milestone: Corpus + tokenizer foundation

The first implementation milestone is intentionally small and reproducible:

1. Define a provenance-aware corpus schema.
2. Create a tiny project-owned/synthetic seed corpus for pipeline testing.
3. Validate every record.
4. Deterministically split train/validation/test.
5. Build dataset manifests and statistics.
6. Train a custom BPE tokenizer locally.
7. Inspect tokenization and vocabulary coverage.
8. Only after the tokenizer is verified, implement the Transformer.

The seed corpus is a **pipeline smoke-test dataset**, not the final pretraining corpus.

## Capability domains

### Developer
- coding
- debugging
- code review
- refactoring
- test generation
- SQL/data work
- systems/devops
- technical reasoning
- verification

### Venture
- startup analysis
- metric calculation
- cohort analysis
- unit economics
- market analysis
- diligence
- evidence verification

## Core VITO protocol

Understand -> identify evidence -> form a testable hypothesis -> calculate/execute -> compare evidence -> state uncertainty -> propose action -> define verification.

The protocol exposes auditable artifacts, not private chain-of-thought.
