# Stage 3 finite-grid theory appendix

Status: **candidate derivation undergoing automated directional validation**.
Stable claims require the numerical identity, orthogonality, reduction, and
held-out simulation gates recorded by the Stage 3 promotion manifest.

## A. Model, target, and assumptions

Let $O=(X,A,Y)\sim P$, where $A\in\{1,\ldots,K\}$. Define

\[
e_j(x)=P(A=j\mid X=x),\qquad
m_j(x)=E_P(Y\mid A=j,X=x).
\]

For a fixed finite grid \(\mathcal L=\{\lambda_1,\ldots,\lambda_L\}\), let

\[
h_\lambda(x)=g\{e(x);\lambda\},\quad
\eta_\lambda=E_P\{h_\lambda(X)\},\quad
\psi_{k,\lambda}(P)
=\frac{E_P\{h_\lambda(X)m_k(X)\}}{\eta_\lambda}.
\]

SCOVA's initial common-target path uses

\[
H(e)=\left(\sum_{j=1}^K e_j^{-1}\right)^{-1},\qquad
g(e;\lambda)=H(e)^\lambda,\qquad \lambda\in[0,1].
\]

The candidate results use the following assumptions.

1. The observations are i.i.d. from a distribution in a dominated,
   nonparametric model.
2. $E_P(Y^2)<\infty$, and every reported influence coordinate has finite,
   nonzero variance.
3. For the active groups and target region, generalized propensities are
   bounded away from zero strongly enough that the displayed influence
   functions are square-integrable.
4. $g(e;\lambda)$ is positive and continuously differentiable in a
   neighborhood of the relevant simplex points, uniformly over the fixed grid.
5. The grid and contrast family are fixed before observing outcomes and do not
   grow with sample size.
6. Cross-fitted nuisance estimates satisfy the rate and boundedness conditions
   in Section F.
7. A causal interpretation additionally requires consistency, conditional
   exchangeability, and target-specific positivity. These assumptions are not
   empirically certified by SCOVA.

## B. Identification

**Proposition 1 (observed-data identification).** Under assumptions 1–4,
\(\psi_{k,\lambda}\) is an observed-data functional. Under assumption 7 it is
also

\[
\psi_{k,\lambda}=E_{Q_\lambda}\{Y(k)\},\qquad
\frac{dQ_\lambda}{dP_X}(x)=\frac{h_\lambda(x)}{\eta_\lambda}.
\]

**Argument.** Conditional exchangeability and consistency give
\(m_k(x)=E\{Y(k)\mid X=x\}\). Integrating this regression against
\(Q_\lambda\) gives the displayed ratio. Without assumption 7, the ratio
retains its standardized observed-data interpretation but not a causal one.

The target distribution itself depends on $P$ through $e(P)$. Consequently,
consistent estimation of $m_k$ alone cannot generally recover the scientific
path if the generalized propensity model converges to the wrong limit.

## C. Independent numerator and denominator derivatives

Consider a regular parametric submodel $P_\epsilon$ through $P$, with score
$s(O)$. Write the orthogonal score decomposition as

\[
s(O)=s_Y(Y\mid A,X)+s_A(A\mid X)+s_X(X).
\]

Let

\[
N_{k,\lambda}(P)=E_P\{h_\lambda(X)m_k(X)\},\qquad
D_\lambda(P)=E_P\{h_\lambda(X)\}.
\]

### C.1 Outcome channel

Only $m_k$ varies along $s_Y$. Its derivative is represented by

\[
h_\lambda(X)\frac{1(A=k)}{e_k(X)}\{Y-m_k(X)\}.
\]

The denominator has no outcome-channel derivative.

### C.2 Covariate-law channel

Holding the conditional laws fixed, the numerator and denominator derivatives
are represented respectively by

\[
h_\lambda(X)m_k(X)-N_{k,\lambda},\qquad
h_\lambda(X)-D_\lambda.
\]

### C.3 Assignment channel

For every $j$, differentiating the conditional treatment probability gives

\[
\dot e_j(x)
=E_P[\{1(A=j)-e_j(x)\}s_A(A\mid X)\mid X=x].
\]

The chain rule therefore yields separate assignment-channel derivatives

\[
\dot N^{A}_{k,\lambda}
=E_P\left[m_k(X)\sum_{j=1}^K
\partial_j g\{e(X);\lambda\}\dot e_j(X)\right],
\]

and

\[
\dot D^{A}_{\lambda}
=E_P\left[\sum_{j=1}^K
\partial_j g\{e(X);\lambda\}\dot e_j(X)\right].
\]

These expressions are derived before taking the quotient; they do not assume
the implemented influence formula.

### C.4 Quotient rule

Since \(\psi=N/D\),

\[
\dot\psi_{k,\lambda}
=D_\lambda^{-1}\{\dot N_{k,\lambda}
-\psi_{k,\lambda}\dot D_\lambda\}.
\]

Combining the two assignment derivatives leaves

\[
D_\lambda^{-1}E_P\left[
\{m_k(X)-\psi_{k,\lambda}\}
\sum_{j=1}^K\partial_j g\{e(X);\lambda\}\dot e_j(X)
\right].
\]

Define

\[
q_\lambda(O)=\sum_{j=1}^K
\partial_j g\{e(X);\lambda\}\{1(A=j)-e_j(X)\}.
\]

Substitution of the conditional representation of \(\dot e_j\) shows that the
assignment contribution is represented by
\(D_\lambda^{-1}\{m_k(X)-\psi_{k,\lambda}\}q_\lambda(O)\).

## D. Efficient influence function

**Theorem 1 (candidate EIF).** Under assumptions 1–4, a mean-zero gradient of
\(\psi_{k,\lambda}\) in the nonparametric model is

\[
\phi_{k,\lambda}(O)=\eta_\lambda^{-1}\left[
h_\lambda(X)\frac{1(A=k)}{e_k(X)}\{Y-m_k(X)\}
+\{m_k(X)-\psi_{k,\lambda}\}
\{h_\lambda(X)+q_\lambda(O)\}
\right].
\]

The three summands lie in the outcome, covariate, and assignment tangent
spaces, respectively. Their inner product with an arbitrary regular score
equals the pathwise derivative obtained in Section C. In the nonparametric
model this gradient is therefore the candidate canonical gradient, subject to
the numerical pathwise-derivative and orthogonality gates required for promotion.

Mean zero follows from iterated expectation:

\[
E\left[\frac{1(A=k)}{e_k(X)}\{Y-m_k(X)\}\mid X\right]=0,
\quad E\{q_\lambda(O)\mid X\}=0,
\]

and $E[h_\lambda(X)\{m_k(X)-\psi_{k,\lambda}\}]=0$.

## E. Simplex-gradient invariance and analytic gradient

Because \(\sum_j\{1(A=j)-e_j(X)\}=0\), replacing every ambient gradient
coordinate \(\partial_j g\) by \(\partial_j g+c(X)\) leaves \(q_\lambda\)
unchanged. Thus the EIF depends only on the gradient projected onto the simplex
tangent space.

For the common geometric path,

\[
\frac{\partial H(e)}{\partial e_j}=\frac{H(e)^2}{e_j^2},\qquad
\frac{\partial g(e;\lambda)}{\partial e_j}
=\lambda\,g(e;\lambda)\frac{H(e)}{e_j^2}.
\]

The production implementation evaluates the same expressions in log space.
JAX automatic differentiation and adaptive central differences are independent
validation oracles; neither defines the production gradient.

## F. Cross-fitted one-step expansion

For each held-out fold, let \(\hat e\) and \(\hat m\) be trained using only its
complement. SCOVA computes the fold-independent plug-in ratio and adds the
empirical outcome and assignment corrections. Influence rows are centered only
for covariance estimation; centering does not alter the point estimate.

The target expansion is

\[
\hat\psi_{k,\lambda}-\psi_{k,\lambda}
=(\mathbb P_n-P)\phi_{k,\lambda}+R_{k,\lambda}.
\]

On sets where the target normalization and active propensities are bounded
away from zero and $g$ has bounded second derivative, the candidate remainder
bound has the form

\[
\max_{\lambda\in\mathcal L}|R_{k,\lambda}|
\lesssim_P
\|\hat m_k-m_k\|_2\|\hat e-e\|_2
+\|\hat e-e\|_2^2+o_P(n^{-1/2}).
\]

Sufficient rate conditions are therefore

\[
\|\hat m_k-m_k\|_2\|\hat e-e\|_2=o_P(n^{-1/2}),
\qquad
\|\hat e-e\|_2^2=o_P(n^{-1/2}),
\]

together with consistency and the regularity conditions above. Cross-fitting
removes the need for a global Donsker condition but does not remove these rate,
moment, or positivity requirements. The quadratic GPS term is intrinsic to
estimating a propensity-dependent scientific target. If \(\hat e\) converges
to $e^\dagger\ne e$, the estimator may instead converge to the pseudo-path
defined by $g(e^\dagger;\lambda)$.

## G. Fixed-grid joint inference

Let $C$ be the fixed declared contrast matrix and stack
\(C\phi_\lambda\) over all contrasts and \(\lambda\in\mathcal L\). Under
assumptions 1–6 and consistent variance estimation,

\[
\sqrt n(\hat\theta-\theta)\rightsquigarrow N(0,\Sigma)
\]

in a finite-dimensional Euclidean space. Conditional centered Gaussian
multipliers applied to the estimated influence rows consistently approximate
the distribution of the maximum absolute studentized coordinate. One
multiplier draw is shared across the complete contrast-by-grid process.

This is a finite-grid result. It does not establish increasing-grid or
continuum-uniform coverage, and it does not justify selecting the grid or
reported contrast family after inspecting outcomes.

## H. Certificate coverage

On the simultaneous-band coverage event, every grid point whose lower band is
positive has a positive true contrast, and every point whose upper band is
negative has a negative true contrast. Sign sets and same-sign suffixes are
deterministic functions of this joint event, so their family-level false
certification probability is bounded by the band's error probability.

For stability, SCOVA forms the joint difference process

\[
\Delta_c(\lambda)=\theta_c(\lambda)-\theta_c(1)
\]

and bootstraps it directly. The reported upper bound for
\(\max_\lambda|\Delta_c(\lambda)|\) is not obtained by comparing separate
marginal intervals.

## I. Reductions

- At \(\lambda=0\), $h_0=1$ and \(\partial_j g(e;0)=0\). The target,
  estimator, and influence function reduce exactly to fixed-target AIPW.
- With two groups, $H(e)=\{e^{-1}+(1-e)^{-1}\}^{-1}=e(1-e)$. At
  \(\lambda=1\), this is the binary overlap population.
- For a fixed tilt independent of $P$, every propensity derivative is zero,
  so $q_\lambda=0$.
- Pairwise and subset paths use the same derivation after restricting $H$ to
  the declared active propensity coordinates. They define different target
  populations and cannot be silently pooled in one global test.

## J. Provenance, implementation map, and nonclaims

The term-by-term theory-to-code correspondence is recorded in
[`stage3_notation_map.md`](stage3_notation_map.md). Numerical validation must
cover the derivative identity, nuisance orthogonality, simplex shifts, endpoint
reductions, finite-difference checks, and JAX checks before stable promotion.

The candidate scope is limited to i.i.d. continuous outcomes, finite declared
grids and contrast families, cross-fitted nuisance learners, and built-in
smooth tilts. It does not cover adaptive lambda selection, continuum bands,
clustered or longitudinal data, interference, custom tilts, TMLE, graphs, or
sensitivity surfaces. Public documentation must make no claim stronger than the
result supported by the directional validation level.
