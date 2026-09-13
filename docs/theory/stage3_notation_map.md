# Stage 3 theory-to-code notation map

This map connects the symbols in the [finite-grid theory appendix](stage3_appendix.md)
to the implementation. The theory uses one-based group indices; NumPy arrays
and package code use zero-based indices. These names are implementation
locations for the current experimental API, not a promise that internal module
paths are stable.

| Theory | Meaning | Implementation |
|---|---|---|
| $e_j(X)$ | generalized propensity | `propensity[:, j]` |
| $m_k(X)$ | outcome regression | `outcome_regression[:, k]` |
| $h_\lambda$ | path tilt | `geometric_tilt_and_gradient(...)[0]` |
| $\partial_j g$ | ambient gradient | `geometric_tilt_and_gradient(...)[1]` |
| $q_\lambda$ | estimated-target correction | `q` in `fit_path` |
| $\eta_\lambda$ | tilt normalization | `denominator / n` |
| $\hat\psi^{\mathrm{plug}}$ | tilted plug-in | `plug_in` |
| $\hat\psi$ | corrected one-step path | `group_means` |
| $\hat\phi$ | centered EIF tensor | `influence` |
| $c^\mathsf{T}\hat\psi$ | contrast path | `ContrastPathResult.estimates` |
| multiplier maximum | joint grid statistic | `SCOVAPathResult.infer` |
| target drift | population shift | `DriftProfile` |
| gate decision | reliability classification | `GateDecision` |

Key source locations are `src/scova/experimental/tilts.py` for the tilt and
analytic gradient, `src/scova/experimental/path.py` for declarations, fitted
paths, and path inference, and `src/scova/experimental/gates.py` for
reliability decisions. The public entry point is `scova.experimental`; inspect
the serialized result rather than depending on private helper functions.

The implementation uses zero-based group and array indices; theoretical group
indices are one-based. `denominator` is the sample sum of the tilt, whereas
$\eta_\lambda$ is its population mean. The stored `influence` tensor is
empirically centered after point estimation and has axes observation × lambda ×
group.
