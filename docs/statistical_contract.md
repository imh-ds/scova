# Fixed-target statistical contract

The first SCOVA milestone targets the observed study covariate distribution,
so the target tilt is fixed at \(h(x)=1\). For group \(k\),

\[
\psi_k = E\{m_k(X)\}, \qquad
m_k(x)=E(Y\mid A=k,X=x), \qquad e_k(x)=P(A=k\mid X=x).
\]

The cross-fitted AIPW estimator averages

\[
Z_{ik}=\widehat m_k(X_i)+
\frac{1(A_i=k)}{\widehat e_k(X_i)}
\{Y_i-\widehat m_k(X_i)\}.
\]

Its estimated influence value is \(\widehat\phi_{ik}=Z_{ik}-\widehat\psi_k\).
SCOVA estimates the covariance of \(\widehat\psi\) with the sample covariance
of the influence rows divided by \(n\). For a declared zero-sum contrast
\(c\), \(\widehat\theta_c=c^T\widehat\psi\), its influence values are
\(\widehat\Phi c\), and pointwise Wald inference uses the corresponding
diagonal of \(c^T\widehat\Sigma c\).

The descriptive interpretation requires consistency of the observed
conditional means, i.i.d. sampling, and support for every requested group on
the study target. A causal interpretation additionally requires consistency,
conditional exchangeability, positivity, no interference, and pre-exposure
covariates. These causal assumptions are declared by the analyst; SCOVA cannot
verify them.

This milestone never silently clips propensity predictions, trims rows,
changes the target, or extrapolates. Missing or non-finite analysis values,
invalid probability vectors, unsupported groups, and invalid contrasts are
errors rather than implicit analytic choices.

## Simultaneous inference

For a finite fitted family, SCOVA stacks the contrast influence values and uses
centered Gaussian multipliers. Each draw records the maximum absolute
studentized statistic over the complete family. The empirical quantile gives
simultaneous two-sided intervals, and exceedance probabilities use the finite
bootstrap correction \((1+r)/(B+1)\). The default is 1,999 draws at 95%
confidence, seeded from the analysis declaration and evaluated in bounded
batches.

The max-t global test addresses the intersection null that every contrast in
the recorded family is zero. The Wald omnibus test uses the Moore–Penrose
inverse of the contrast covariance; its chi-squared degrees of freedom equal
the numerical rank, so redundant all-pairwise families do not inflate the
reference dimension. These are asymptotic procedures and do not repair weak
support, nuisance-model failure, or unmeasured confounding.

Interpretation and computational inference status are distinct. A causal
declaration remains `exploratory-only` until later SCOVA stages implement and
pass declared design gates. Simultaneous inference reports `complete`,
`warning`, or `refused` independently of that interpretation.
