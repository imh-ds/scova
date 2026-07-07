# Estimated-tilt influence function and release gate

This note fixes the statistical contract for SCOVA's experimental finite-grid
overlap path. It is an implementation derivation, not yet an externally
reviewed theorem. Consequently the API remains under `scova.experimental`.

## Parameter

For a smooth tilt \(h(X)=g(e(X);\lambda)\), define

\[
\psi_k(\lambda)=\frac{E\{h(X)m_k(X)\}}{\eta_\lambda},
\qquad \eta_\lambda=E\{h(X)\}.
\]

The path currently uses \(h_\lambda=(K h_{ow})^\lambda\), where
\(h_{ow}=(\sum_j e_j^{-1})^{-1}\). The factor \(K^\lambda\) is constant
within a target and therefore cancels from its normalized distribution.

## Propensity channel

For a regular observed-data submodel with score \(s(O)\),

\[
\dot e_j(x)=E[\{1(A=j)-e_j(x)\}s(O)\mid X=x].
\]

Applying the quotient rule to the propensity channel gives

\[
\frac{1}{\eta_\lambda}E\left[
\{m_k(X)-\psi_k(\lambda)\}
\sum_j \partial_jg(e(X);\lambda)\dot e_j(X)
\right].
\]

Thus the observed-data gradient contribution is obtained with

\[
q_\lambda(O)=\sum_j\partial_jg(e(X);\lambda)
\{1(A=j)-e_j(X)\}.
\]

Because the multinomial residuals sum to zero, adding the same scalar to every
coordinate of the ambient gradient leaves \(q_\lambda\) unchanged.

Combining the outcome, covariate-distribution, and propensity channels gives

\[
\phi_{k,\lambda}(O)=\eta_\lambda^{-1}\left[
h_\lambda\frac{1(A=k)}{e_k}(Y-m_k)
+(m_k-\psi_k)(h_\lambda+q_\lambda)
\right].
\]

The implementation uses the corresponding one-step estimator around the
tilted plug-in mean and empirically centers the evaluated influence values only
for covariance estimation. At \(\lambda=0\), \(h=1\), the gradient and
\(q\) vanish and the estimator reduces exactly to SCOVA's fixed-target AIPW
estimator.

For binary treatment at \(\lambda=1\), the target tilt is proportional to
\(e(1-e)\), so the estimand and influence-function structure reduce to the
overlap-population special case. The experimental one-step point estimate need
not be numerically identical in finite samples to a particular parametric
propensity-score weighted estimator; the reduction is an estimand and
first-order influence-function statement.

## Numerical gates

The following checks are automated:

- analytic gradients versus central finite differences;
- optional analytic-versus-JAX gradients;
- invariance to a constant simplex-gradient shift;
- exact multinomial submodel identity \(d\psi/d\epsilon=E(\phi s)\);
- second-order corrected nuisance remainder versus a first-order naive
  ablation;
- fixed-target and binary-overlap endpoint reductions;
- simultaneous coverage over a declared finite grid.

External derivation review and continuum-uniform theory remain incomplete.
SCOVA therefore makes no continuum confidence-band claim and does not promote
this module to the stable namespace.
