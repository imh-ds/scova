# Stage 3 positioning

SCOVA does not claim that balancing weights, overlap populations,
multi-treatment overlap weights, smooth trimming, or multiplier bands are new.
The Stage 3 contribution is their integration into a single declared target
process with estimated-target correction and population-aware certificates.

| Method | Target object | Inference object | Population interpretation |
|---|---|---|---|
| Fixed AIPW | One study target | Pointwise/fixed-family | Scientific population only |
| Generalized overlap weighting | One equipoise target | Endpoint contrasts | Overlap population only |
| Smooth trimming | One smoothed trimmed target | Target-specific interval | Smoothed retained population |
| SCOVA Stage 3 | Declared study-to-overlap grid | Joint contrast × grid bands | Effect path paired with target drift and stability certificates |

The distinctive outputs are:

1. the exact grid on which family-wise inference is controlled;
2. sign sets and suffix certificates that do not assume path monotonicity;
3. a simultaneous upper bound on effect drift from the overlap endpoint; and
4. covariate, ESS, concentration, and propensity-simplex descriptions of each
   target population.

The initial default is a common K-way path. Pairwise and subset paths are
separate analyses and cannot be combined into a common-target global claim.

