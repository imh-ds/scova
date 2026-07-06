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

