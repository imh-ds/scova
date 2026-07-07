# SCOVA

**Support-aware Covariate Overlap and Variance Analysis**

SCOVA is an early-stage Python methodology and package for honest comparisons
among naturally occurring groups. The current `0.2.0` milestone implements a
fixed-study-population (`h(x) = 1`), cross-fitted, multi-group AIPW estimator
for continuous outcomes with simultaneous finite-family inference.

## Status and scope

The current vertical slice provides:

- immutable, hashed analysis declarations;
- deterministic stratified cross-fitting;
- multinomial propensity and group-specific outcome models;
- standardized group means, pairwise/custom contrasts, influence values, and
  pointwise Wald inference;
- deterministic Gaussian multiplier-bootstrap intervals, max-t-adjusted
  p-values, and max-t and rank-aware Wald global tests;
- initial overlap, balance, effective-sample-size, and influence diagnostics;
- deterministic simulation fixtures with oracle nuisances; and
- versioned result persistence without pickle.

Overlap paths, simultaneous inference, TMLE, comparability graphs, partial
identification, and sensitivity surfaces are intentionally deferred. See
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
