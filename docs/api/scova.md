# Fixed-target SCOVA public API

SCOVA estimates standardized group means for one fixed study population. The
target is the observed eligible covariate distribution, not a population chosen
after fitting and not a population created by trimming or matching.

The normative estimand, assumptions, and inference rules are in the
[fixed-target statistical contract](../contracts/statistical_contract.md).

## Quick start

```python
from scova import SCOVA, SCOVADeclaration
from scova.simulate import generate_data

simulation = generate_data("observational", n=1_000, seed=42)
declaration = SCOVADeclaration(
    outcome="outcome",
    group="group",
    covariates=("x1", "x2", "x3"),
    interpretation="descriptive",
    n_splits=5,
    random_state=42,
)

result = SCOVA().fit(simulation.data, declaration)
inference = result.infer()

print(dict(zip(result.group_labels, result.group_means, strict=True)))
print(inference.to_dict())
```

`SCOVA().fit(data, declaration)` accepts a pandas `DataFrame`. The declared
outcome and covariate columns must be numeric, finite, and complete. The group
column must contain at least two groups, and every group must have enough rows
for the declared number of cross-fitting folds. Invalid input is rejected with
an exception; SCOVA does not silently repair the analysis.

## Declaration and nuisance models

`SCOVADeclaration` records the outcome, group, baseline covariates,
interpretation, cross-fitting policy, random seed, and optional named
zero-sum contrasts. Declarations are immutable and hashed, so the result can
be tied back to the exact requested analysis.

`SCOVA()` defaults to `nuisance_strategy="adaptive"`. It always uses the
flexible histogram-gradient-boosting propensity learner and selects between a
regularized linear and boosting outcome learner by deterministic inner-fold
loss. Use `nuisance_strategy="linear"` for the transparent Ridge/logistic
baseline, or use `nuisance_strategy="custom"` with both compatible nuisance
models supplied.

## Results and inference

`SCOVAResult` contains standardized group means, automatically generated
pairwise contrasts, declared custom contrasts, influence values, covariance,
cross-fitting assignments, nuisance metadata, and diagnostics.

- `result.contrast(weights)` computes a pointwise Wald contrast without
  refitting the nuisance models.
- `result.infer()` applies multiplier-bootstrap max-t inference to the exact
  finite contrast family recorded in the result.
- `result.save(path)` and `SCOVAResult.load(path)` persist versioned,
  non-pickle result artifacts.

Point estimates and computational inference status are separate from causal
interpretation. Descriptive declarations return `descriptive-only`; causal
declarations return `exploratory-only`. Diagnostics and intervals do not prove
exchangeability, positivity, absence of unmeasured confounding, or causal
validity in an applied dataset.

## Stable versus experimental APIs

The fixed-target estimator and its result/inference objects are the current
public SCOVA surface. Estimated-target overlap paths, outcome-free graph
design, and bounded-outcome anchor APIs are experimental and must be imported
from `scova.experimental`. They remain subject to separate evidence gates and
may change without the fixed-target API's compatibility guarantees.
