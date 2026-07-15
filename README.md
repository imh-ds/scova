# SCOVA

**Support-aware Covariate Overlap and Variance Analysis**

SCOVA is an early-stage Python methodology and package for honest comparisons
among naturally occurring groups. The stable `0.2.0` core implements a
fixed-study-population (`h(x) = 1`), cross-fitted, multi-group AIPW estimator
for continuous outcomes with simultaneous finite-family inference.

The `0.3.0.dev0` source tree also contains an experimental finite-grid smooth
overlap path. It remains experimental pending the frozen directional calibration,
held-out validation, and pinned-gradient runs. Its production-stabilization machinery and remaining
release blockers are tracked in
[`docs/stage3_stabilization.md`](docs/stage3_stabilization.md). The executable
campaign sequence is in the
[`Stage 3 directional runbook`](docs/stage3_directional_runbook.md).
The manually triggered `Stage 3 directional release validation` GitHub Actions
workflow performs the complete sharded validation and evidence aggregation.

## Status and scope

The current vertical slice provides:

- immutable, hashed analysis declarations;
- deterministic stratified cross-fitting;
- adaptive cross-fitted nuisance learning: regularized linear and
  histogram-gradient-boosting candidates selected by inner-fold loss;
- multinomial propensity and group-specific outcome models;
- standardized group means, pairwise/custom contrasts, influence values, and
  pointwise Wald inference;
- deterministic Gaussian multiplier-bootstrap intervals, max-t-adjusted
  p-values, and max-t and rank-aware Wald global tests;
- initial overlap, balance, effective-sample-size, and influence diagnostics;
- deterministic simulation fixtures with oracle nuisances; and
- versioned result persistence without pickle.

TMLE, comparability graphs, partial identification, sensitivity surfaces, and
continuum-uniform path inference are intentionally deferred. The finite-grid
overlap path is available experimentally. See
[the methodology plan](scova_methodology_plan.md) and
[statistical contract](docs/statistical_contract.md).

## Installation

SCOVA requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

Install `.[jax]` only when running the optional automatic-gradient validation.

Stage 3 uses provisional diagnostic thresholds until the calibration campaign
is complete. Provisional thresholds may warn or refuse, but they cannot emit a
confirmatory pass. A hard refusal suppresses simultaneous bands, adjusted
p-values, global tests, and sign/stability certificates while retaining
descriptive estimates and target-drift summaries.

## Minimal example

```python
from scova import SCOVA, SCOVADeclaration
from scova.simulate import generate_data

simulation = generate_data("observational", n=1000, seed=42)
declaration = SCOVADeclaration(
    outcome="outcome",
    group="group",
    covariates=("x1", "x2", "x3"),
    n_splits=5,
    random_state=42,
)
result = SCOVA().fit(simulation.data, declaration)
simultaneous = result.infer()

print(dict(zip(result.group_labels, result.group_means, strict=True)))
print(result.contrasts["g0 - g1"])
print(simultaneous.contrast("g0 - g1"))
print(simultaneous.global_test)
print(result.diagnostics["effective_sample_sizes"])
```

A complete three-group workflow is available in
[`examples/three_group.py`](examples/three_group.py).

### Nuisance-learning defaults

`SCOVA()` defaults to `nuisance_strategy="adaptive"`. Within each outer
cross-fitting fold it chooses between regularized linear and
histogram-gradient-boosting nuisance candidates by deterministic inner-fold
log loss for the propensity model and mean squared error for each group outcome
model. The result artifact records the selected learner and candidate scores.

Use `SCOVA(nuisance_strategy="linear")` for the former transparent
Ridge/logistic baseline. Use `nuisance_strategy="custom"` together with both
`propensity_model` and `outcome_model` to supply compatible scikit-learn
estimators.

An experimental target-path workflow is available in
[`examples/overlap_path.py`](examples/overlap_path.py). It uses
`PathDeclaration` and `fit_path` from `scova.experimental`, jointly controls a
declared 21-point grid, and reports sign, stability, and target-drift output.
The mathematical contract and novelty boundary are documented in
[`estimated_tilt_eif.md`](docs/estimated_tilt_eif.md) and
[`stage3_positioning.md`](docs/stage3_positioning.md).

## Interpretation

The default verdict is `descriptive-only`. Selecting a causal interpretation
records the analyst's assumption claim and produces `exploratory-only`; it does
not make exchangeability or positivity empirically verifiable. `certified` is
reserved for a future declared-gate system. SCOVA never silently clips
propensity scores, trims observations, or changes the target population.

`result.infer()` controls family-wise error over the exact fitted contrast
family recorded in its return value. Pointwise intervals on `ContrastEstimate`
and simultaneous intervals on `SimultaneousContrastResult` are intentionally
separate. The max-t global test rejects when at least one family contrast is
nonzero; the Wald test is an omnibus quadratic-form test whose degrees of
freedom use the numerical rank of the contrast covariance.

## License

SCOVA is licensed under GPL-3.0.
