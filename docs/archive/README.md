# Local documentation archive

This is a local-only archive of historical, experimental, and planning
documents that are already consolidated into the public
[methodology decision log](../methodology_decision_log.md). The archive index
is tracked so a maintainer can find the material after a checkout; the files
under its category folders are ignored and are not part of the public package
documentation.

| Category | Retained material |
| --- | --- |
| [`historical/qualification/`](historical/qualification/) | Retired SCOVA-CF qualification and calibration documents, including the observational workflow |
| [`historical/reference-protocols/`](historical/reference-protocols/) | Superseded reference validation and recalibration runbooks, reports, and workflows |
| [`historical/comparative-studies/`](historical/comparative-studies/) | Completed comparative-methods study and inference-audit source documents |
| [`experimental/`](experimental/) | Stage 4 campaign and Stage 5B promotion-audit documents |
| [`planning/`](planning/) | Internal study plans and specifications |

The original-to-archive mapping is:

| Original public path | Local archive path |
| --- | --- |
| `docs/historical/scova_cf_methodological_contract_v1.md` | `historical/qualification/scova_cf_methodological_contract_v1.md` |
| `docs/historical/scova_cf_qualification_overview_v1.md` | `historical/qualification/scova_cf_qualification_overview_v1.md` |
| `docs/historical/workflows/cf-observational-qualification.yml` | `historical/qualification/workflows/cf-observational-qualification.yml` |
| `docs/historical/workflows/cf-reference-v10-validation.yml` | `historical/reference-protocols/cf-reference-v10-validation.yml` |
| `docs/historical/workflows/cf-reference-v11-validation.yml` | `historical/reference-protocols/cf-reference-v11-validation.yml` |
| `docs/cf_reference_*.md` and `docs/scova_cf_*reference_runbook.md` | `historical/reference-protocols/` |
| `docs/scova_cf_observational_calibration_v2.md` | `historical/qualification/observational_calibration_v2.md` |
| `docs/scova_cf_comparative_*.md` | `historical/comparative-studies/` |
| `docs/stage4_campaign_runbook.md`, `docs/stage5b_promotion_audit.md` | `experimental/` |
| `docs/superpowers/plans/*.md`, `docs/superpowers/specs/*.md` | `planning/` |

The former paths were reorganized here without deleting the local files. Git
history remains available for every previously tracked document, while this
archive preserves an easy local browsing path. Use the decision log and the
current public contracts as release authority; do not use archive material to
authorize a new claim.
