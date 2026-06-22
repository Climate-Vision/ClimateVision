# Team Role Reassignment

**Effective date:** 2026-06-22

## Departure

**Hopelyn** (`@Hopelynconsult`) has stepped away from the ClimateVision project. We thank her for her contributions, including the calibration-metrics work merged in PR #51.

### Security cleanup

Her internal credentials file (`Hopelyn_Role_INTERNAL.md`) has been removed from the project. The associated personal access token must be revoked immediately by the project owner to prevent unauthorized access.

> **Action required:** Revoke the PAT ending in `y4OfP1h` in the GitHub account security settings for `@Hopelynconsult`.

---

## Reassigned responsibilities

Hopelyn’s former role was **Responsible AI & Model Evaluation Lead**. The ownership areas have been redistributed as follows:

### `@obielin` (Linda Oraegbunam) — Governance & Evaluation Engineering

Already the primary author of the governance modules (PRs #35–39), Linda now owns the implementation and maintenance of:

- Automated model-card generator (`src/climatevision/governance/model_card.py`)
- Regional bias / fairness audit framework (`src/climatevision/governance/bias_audit.py`)
- Hash-chained audit logger (`src/climatevision/governance/audit_logger.py`)
- Anomaly detection for inference outputs (`src/climatevision/governance/anomaly_detector.py`)
- Release CI gate for metrics, fairness, and security

### `@Goldokpa` (Gold Okpa) — Responsible AI Policy & Oversight

As project owner, Gold now owns the policy and process layer:

- Responsible-AI policy and contributor-facing governance checklist
- PR review gate for any PR touching `src/climatevision/models/` or `src/climatevision/inference/`
- Calibration and uncertainty-quantification roadmap
- Human-in-the-loop review process for low-confidence detections
- Production drift-monitoring roadmap

### Ongoing collaboration

- Linda implements the tooling; Gold defines the policies and enforces the review checklist.
- All model/inference PRs must receive approval from `@Goldokpa` and, when governance tooling is affected, from `@obielin`.

---

## Files affected by this change

- `team_docs/Hopelyn_Role_INTERNAL.md` — **deleted**
- `team_docs/ROLE_REASSIGNMENT.md` — **created** (this file)
- `README.md` — Contributors table updated
- `CHANGELOG.md` — Team change recorded under `[Unreleased]`
