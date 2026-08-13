# Regular SCOVA simulation evidence

This folder is a convenient, versioned copy of the compact release artifacts already used by the regular SCOVA examples.

## Included results

- `simulated-nonlinear-overlap.json`: in a common-support nonlinear simulation with a true contrast of `+3.0`, flexible cross-fitted SCOVA/AIPW estimated `2.93` (95% CI `2.76` to `3.10`), while a deliberately misspecified linear ANCOVA estimated `5.46`.
- `simulated-support-failure.json`: with disjoint covariate support and a true direct contrast of zero, the graph-support logic refused the contrast rather than extrapolating; the linear ANCOVA estimate was `20.61`.
- `simulated-program-tracks.json`: three-group illustrative results contrasting raw ANOVA, linear ANCOVA, fixed-target SCOVA, and experimental bounded/transport anchors.
- `stage3_promotion.json`: directional validation/promotion manifest for the finite-grid overlap path.
- `stage5b_promotion_audit.json` and `stage5b-lipschitz-anchor-evidence.json`: reproducibility evidence for the experimental Stage 5B bounded/transport anchor.

## Limits

The finite-grid, graph, and transport paths remain experimental. Their preserved artifacts demonstrate the stated simulated behavior and engineering checks; they do not turn untestable causal or extrapolation assumptions into facts. See [the stabilization status](../../../docs/stage3_stabilization.md) and [the Stage 5B audit](../../../docs/stage5b_promotion_audit.md).
