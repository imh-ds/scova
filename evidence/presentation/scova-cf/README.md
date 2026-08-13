# SCOVA-CF comparative simulation evidence

These are separate, completed descriptive methods studies of the current SCOVA-CF estimator. They compare eligible-population ATE estimators in frozen two-group simulated data with five covariates. Propensity-score matching is reported separately because it estimates a retained-treated ATT.

## Completed evidence sets

| Version | Final Actions run | Design | Artifact location |
| --- | --- | --- | --- |
| v2 | [31575961864](https://github.com/imh-ds/scova/actions/runs/31575961864) | Linear/interaction outcomes × linear/nonlinear assignment × adequate/poor overlap; 1,000 replications per cell | [`v2-final/`](v2-final/) |
| v3 | [31637266873](https://github.com/imh-ds/scova/actions/runs/31637266873) | Smooth-nonlinear/threshold outcomes × smooth-nonlinear/threshold assignment × adequate/poor overlap; 1,000 replications per cell | [`v3-final/`](v3-final/) |

Both complete artifacts contain eight cells, 1,000 replications per cell, and 48,000 estimator records. v2 checksum: `fe7fa979847961fea2756c98ce697cca84be2a108aa62a8de1be34060a603ef5`. v3 checksum: `dffa3adceef89c8ba4fbfdbf7ebe93c2ed8b7f953a1b7b0105884b55911438ea`.

The v2 Markdown report is the audited rendering: it preserves the frozen JSON but omits invalid PSM/DRLearner coverage comparisons. The original unaudited renderer remains in the local Actions download only so it cannot be mistaken for the corrected report.

## Interpretation

SCOVA-CF was stable and near-unbiased in both frozen evidence sets. It was particularly strong when linear functional-form assumptions were wrong. These results are descriptive performance evidence within the named DGPs only. They neither certify observational causal identification nor establish that all AIPW/DR learners behave similarly in applied data.

Coverage is interpretable only for SCOVA-CF, independent AIPW, and model-based ANCOVA. The PSM and DRLearner adapters do not provide valid ATE/ATT sampling standard errors in these studies, so their coverage fields are not a comparison metric. See [the inference audit](../../../docs/scova_cf_comparative_inference_audit.md).
