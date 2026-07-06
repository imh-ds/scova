# SCOVA

**Support-aware Covariate Overlap and Variance Analysis**

SCOVA is an early-stage Python methodology and package for honest comparisons
among naturally occurring groups. The current `0.1.0` milestone implements a
fixed-study-population (`h(x) = 1`), cross-fitted, multi-group AIPW estimator
for continuous outcomes.

## Status and scope

The current vertical slice provides:

- immutable, hashed analysis declarations;
- deterministic stratified cross-fitting;
- multinomial propensity and group-specific outcome models;
- standardized group means, pairwise/custom contrasts, influence values, and
  pointwise Wald inference;
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

print(dict(zip(result.group_labels, result.group_means, strict=True)))
print(result.contrasts["g0 - g1"])
print(result.diagnostics["effective_sample_sizes"])
```

A complete three-group workflow is available in
[`examples/three_group.py`](examples/three_group.py).

## Interpretation

The default verdict is `descriptive-only`. Selecting a causal interpretation
records the analyst's assumption claim; it does not make exchangeability or
positivity empirically verifiable. SCOVA never silently clips propensity
scores, trims observations, or changes the target population.

## License

SCOVA is licensed under GPL-3.0.

