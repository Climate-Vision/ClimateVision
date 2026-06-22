# Changelog

All notable changes to ClimateVision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- SECURITY.md — private vulnerability reporting via GitHub Security Advisories
- CODEOWNERS — automatic review assignment to @Goldokpa
- Pull request template for structured contributor guidance
- Dependabot configuration for pip, npm, and GitHub Actions updates
- CHANGELOG.md (this file)
- CITATION.cff for GitHub "Cite this repository" button
- `team_docs/ROLE_REASSIGNMENT.md` documenting departure of @Hopelynconsult and redistribution of Responsible AI responsibilities to @Goldokpa and @obielin

### Removed
- `team_docs/Hopelyn_Role_INTERNAL.md` (internal credentials file for departed team member); associated PAT must be revoked

### Changed
- CODE_OF_CONDUCT.md — replaced placeholder email with GitHub private reporting link

### Removed
- SETUP_COMPLETE.md — internal artifact moved out of public repo
- team_docs/ — internal role documents moved out of public repo

---

## [0.2.0] — 2026-03-04

### Added
- FastAPI REST backend with paginated run history and stats endpoint
- React dashboard with interactive bbox map, Recharts analytics, and confidence gauges
- U-Net semantic segmentation for deforestation and arctic ice detection
- Siamese network change detection
- Google Earth Engine integration with cloud masking and 256×256 tiling
- MLflow experiment tracking
- ONNX model export
- Flood detection analysis type
- NGO management — organisation registration, region subscriptions, email/webhook alerts
- Full OpenAPI docs at `/docs`

### Changed
- README rewritten to concise FastAPI-style format

---

## [0.1.0] — 2026-03-04

### Added
- Initial repository structure and governance files
- Basic project scaffold (src layout, config, notebooks, scripts)
- MIT License
- Contributing guide and Code of Conduct

[Unreleased]: https://github.com/Climate-Vision/ClimateVision/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Climate-Vision/ClimateVision/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Climate-Vision/ClimateVision/releases/tag/v0.1.0
