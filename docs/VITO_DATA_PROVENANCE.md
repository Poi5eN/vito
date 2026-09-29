# VITO Evidence and Provenance Standard v0.1

Every startup-analysis observation should be traceable.

## Required evidence fields

- `evidence_id`
- `startup_id`
- `claim`
- `value`
- `unit`
- `source_url`
- `publisher`
- `source_type`
- `reliability_tier`
- `published_at`
- `retrieved_at`
- `as_of`
- `source_excerpt_or_locator`
- `verification_status`

## Verification states

- `verified`: supported by a reliable source or reproducible calculation
- `partially_verified`: some inputs supported; some uncertainty remains
- `claimed`: supplied by the company/founder but not independently verified
- `conflicting`: credible sources disagree
- `unverified`: insufficient evidence
- `calculated`: derived deterministically from documented inputs

## Reproducibility

An analysis run should record:

- dataset snapshot/version
- source snapshot metadata
- model version
- scoring methodology version
- tools used
- code revision
- run timestamp

The final report must be reconstructable from the stored evidence and calculation records.
