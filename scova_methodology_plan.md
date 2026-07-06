# SCOVA: Support-aware Covariate Overlap and Variance Analysis

**A consolidated review of two proposed frameworks, and a revised, theoretically ambitious methodology plan for ML-assisted group means comparison — written as a development roadmap.**

---

# Part I — Review and Consolidation of the Two Source Frameworks

## 1. What each framework gets right

The two documents — **HOTC** (Honest Overlap-Targeted Contrasts) and **SCGC** (Standardized Counterfactual Group Contrasts) — converge on the same correct diagnosis and roughly 70% of the same architecture. That convergence is itself informative: it marks the parts of the design space that are settled, and it sharpens where the remaining novelty must come from.

**Shared, correct core (retain without reservation):**

1. The problem is an **estimand-design problem before it is an estimation problem**. A t-test compares observed marginal means; for naturally occurring groups the hard question is *which standardized or counterfactual contrast is even identified, for which population*.
2. ML belongs in the **nuisance layer only** (generalized propensity scores $e_k(x)$, outcome regressions $m_k(x)$), consumed through **Neyman-orthogonal scores with cross-fitting**, never as a counterfactual oracle. "Fit ML, impute counterfactuals, t-test the imputations" is correctly identified as fatally flawed by both.
3. **Overlap is estimand-defining, not a robustness footnote.** Both adopt tilted target populations (ATE-type, ATO-type, K-way overlap) and both treat positivity failure as a reason to change or refuse the estimand, transparently.
4. **Simultaneous inference over a contrast family** via multiplier bootstrap on estimated efficient influence functions (EIFs) is the right inferential backbone for $K \ge 2$.
5. A **dual interpretation regime**: causal when group membership admits a defensible intervention interpretation; *standardized descriptive* otherwise. Same functional, different license.
6. **Refusal rules** and reliability outputs (effective sample size, influence concentration, extrapolation flags) as first-class outputs.
7. Both correctly refuse the "successor to the t-test" marketing and correctly enumerate what is *not* novel (AIPW/DML/TMLE, GPS, overlap weights, causal forests, BART, conformal).

**Distinctive strengths of HOTC (keep):**

- The **contrast-specific target calculus**: pairwise targets, subset targets, and the recognition that a pairwise contrast can be estimable when the K-way comparison is not.
- The **comparability graph**: groups as nodes, design-stage overlap-certified contrasts as edges, cliques as jointly comparable subsets. This is the single best structural idea in either document for the multi-group setting, and it directly attacks the ANOVA-shaped hole in current practice ("global F-test then post-hocs" presumes all K groups are jointly comparable, which observational data routinely violates).
- The **overlap path** $h_\lambda$ and "estimand path bands" (its Contribution C), flagged but left undeveloped.
- The **two-stage design/analysis firewall**: target and contrast family selected from $(X, A)$ only, outcomes touched only afterward.
- The most complete diagnostics suite (propensity simplex, ESS, influence concentration, density-ratio tails, estimator triangulation) and the most useful simulation scenarios (Scenario 5: pairwise overlap without K-way overlap; Scenario 7: learned contrast under global null).

**Distinctive strengths of SCGC (keep):**

- Making **valid selective inference for the data-adaptive "winner" contrast** ($\Delta^\star = \max_{j \ne k}(\theta_j - \theta_k)$) the theoretical anchor. This is the right instinct: "which group is best" is the question applied users actually ask, and naive inference on it is invalid (winner's curse). A clean EIF-based treatment of it is a genuine contribution.
- **Centering sensitivity analysis to unmeasured confounding** as a mandatory headline output rather than an appendix, since unconfoundedness is untestable and ML does nothing for it.
- **Ranking with confidence sets** (confidence statements on ranks / top-$m$ membership) rather than fake point rankings.
- The **per-group, target-aware extrapolation index** — quantifying how far $\hat m_k$ is evaluated outside group $k$'s own support when projected onto the shared target.
- The most ruthless novelty audit (its Section J table), which any serious plan must pass.

## 2. Shared blind spots — the gaps the new methodology must fill

Both documents, read as a pair, leave seven substantive holes. These define where the genuinely new work lives.

**Gap 1 — No rigorous treatment of estimated targets.** Both plug $\hat h(x) = g(\hat e(x))$ into the fixed-$h$ influence function and wave at "bootstrap or conditional inference." But when the target tilt depends on estimated propensities, the efficient influence function acquires a **correction term from the pathwise derivative of the tilt with respect to $e$**. For binary-treatment overlap weights this term is known to have special structure (Li–Morgan–Zaslavsky's "happy coincidence"); for general smooth multi-group tilts, and *uniformly along a tilt path*, it has not been packaged. Ignoring it yields wrong standard errors precisely in the regime the framework advertises (overlap-adaptive targets). This is a real derivation, not bookkeeping, and it is the price of admission for everything else.

**Gap 2 — The estimand path is decorative, not inferential.** HOTC proposes $\theta_c(h_\lambda)$ as a "diagnostic"; SCGC proposes reporting "the estimand curve over $\eta$." Neither treats the path as what it should be: the **primary inferential object** — a function-valued estimand with a uniform functional central limit theorem, simultaneous confidence bands over $(contrast \times \lambda)$, and *derived functionals with valid inference* (the set of tilts on which the sign is certified, the minimal tightening at which a contrast becomes estimable). Elevating the path from sensitivity plot to estimand process is the conceptual leap that neither document completes.

**Gap 3 — Both change the question and never answer the original one.** Overlap targeting is honest about estimating a different estimand, but the applied user's question ("what about the *whole* population?") is abandoned rather than bounded. The missing piece is **partial identification of the full-population contrast**: decompose it into an identified overlap component and a non-overlap remainder, bound the remainder under explicit, tunable extrapolation assumptions (bounded outcomes; Lipschitz/Hölder extrapolation budgets), and deliver Imbens–Manski-type confidence intervals for the resulting identified set. This converts "we refuse" into "here is exactly what the data can and cannot say about your original question, as a function of how much extrapolation you are willing to assume." Neither document contains this, and it neutralizes their own strongest self-objection ("you are solving positivity by changing the question").

**Gap 4 — Sensitivity, overlap, and extrapolation are treated as separate appendices rather than a single geometry.** Unmeasured confounding (SCGC's marginal sensitivity model), support/extrapolation (both), and target tilting (both) are three *epistemic budget axes* of one object. A unified reporting object — a robustness profile over (tilt $\lambda$, extrapolation budget $\Gamma$, confounding budget $\Lambda$) with certified summaries on each axis — is absent from both and is a distinctive, publishable packaging with real theory on each axis.

**Gap 5 — Everything leans on the CLT exactly where the CLT is weakest.** Both frameworks' inference is asymptotic-normal via cross-fitted EIFs, then both admit (HOTC Objection 9, SCGC K8) that weak overlap, small effective sample sizes, and heavy-tailed weights break the Gaussian approximation in finite samples. Neither supplies a robust backend. A **doubly robust conditional randomization test** (model-X style: resample group labels from the fitted GPS, recompute the orthogonalized statistic) provides finite-sample-calibrated testing of the sharp adjusted null with a provable level bound in terms of GPS estimation error — precisely the fallback needed when the ESS gate is marginal.

**Gap 6 — Nonsmoothness of hard trimming acknowledged, never solved.** Both note that indicator-based overlap sets $\mathbb{1}\{\min_k e_k(x) \ge \eta\}$ make the estimand boundary nonsmooth and inference awkward, then proceed anyway. The resolution is structural: make **smooth tilts the primitive** (hard trimming as a limit), which simultaneously fixes differentiability for Gap 1, enables the Donsker arguments for Gap 2, and removes the $\eta$-selection researcher degree of freedom by reporting the whole path.

**Gap 7 — No development-grade specification.** Neither document says how the theory becomes software: no module decomposition, no numerical validation strategy for the influence functions, no enforcement mechanism for the design/analysis firewall, no computational plan for path-indexed bootstrap inference. A roadmap "detailed enough for coding to begin" needs all of it.

Two smaller consolidation notes. First, HOTC's list of acceptable inference modes for learned targets (fixed / conditional-on-design / honest split) and SCGC's insistence on stating *which* error rate is controlled (family-wise vs. selective) should be merged into a single explicit inference-mode taxonomy. Second, both simulation plans are good and largely overlapping; the merged plan below keeps HOTC's multi-group structural scenarios and SCGC's "designed to fail" honesty checks and pre-registered win conditions.

## 3. Consolidation decisions

| Component | HOTC | SCGC | Decision |
|---|---|---|---|
| Estimand-first framing, dual causal/descriptive modes | ✓ | ✓ | **Keep**; formalize as an *interpretation ladder* tied to tiered assumptions |
| Orthogonal/DR estimation, cross-fitting, TMLE backend | ✓ | ✓ | **Keep** as estimation engine (explicitly non-novel) |
| Tilted targets ($h$), overlap targets | ✓ | ✓ | **Keep**; smooth tilts become the primitive (Gap 6) |
| Overlap path $h_\lambda$ | sketched | sketched ($\eta$-curve) | **Promote** to primary estimand process with uniform inference (Gap 2) |
| Estimated-tilt EIF correction | absent | absent | **New** (Gap 1) — required theory |
| Comparability graph / hypergraph | ✓ | – | **Keep**; attach conditional-coverage theorem |
| Winner / data-adaptive contrast selective inference | mentioned | ✓ anchor | **Keep** as a core inference module |
| Ranking confidence sets | mentioned | ✓ | **Keep** (adopt inference-on-ranks machinery) |
| Partial identification of the full-population contrast | absent | absent | **New** (Gap 3) — anchored decomposition, extrapolation-budget bounds |
| Sensitivity analysis (marginal sensitivity model) | optional module | mandatory | **Keep mandatory**; integrate along the path (Gap 4) |
| Three-axis robustness profile | absent | absent | **New** (Gap 4) |
| Finite-sample robust backend (DR-CRT) | permutation as diagnostic | permutation listed | **New formalization** (Gap 5) |
| Design/analysis firewall | ✓ | pre-registration | **Keep**; enforce in software (Gap 7) |
| Refusal rules → typed verdicts | ✓ | min-ESS gate | **Keep**; formalize as a verdict type system |
| Diagnostics suite | ✓ (fullest) | ✓ (extrapolation index) | **Merge** |
| Simulation plan | ✓ | ✓ (win conditions) | **Merge** |
| "Successor to the t-test" framing | rejected | rejected | **Rejected**; positioning statement in §18 |

The consolidated method is named **SCOVA** — *Support-aware Covariate Overlap and Variance Analysis*. The name foregrounds the framework's operational commitments: diagnose empirical support before estimation, adjust comparisons across covariate distributions, define overlap-aware target populations, and carry uncertainty from estimation, extrapolation, and confounding into the reported analysis.

---

# Part II — The SCOVA Framework: Conceptual Core

## 4. Thesis

**Classical framing.** "Are the group means different?" — answered by a scalar statistic under exchangeability that observational group membership does not satisfy.

**SCOVA framing.** A comparison among naturally occurring groups is not a scalar but a **process**: a family of standardized contrasts indexed by *explicit epistemic budget parameters*, each of which quantifies one thing the analyst must assume or concede:

- **$\lambda$ — the target-tilt axis (who is being compared).** $\lambda = 0$ is the analyst's scientific target population; $\lambda = 1$ is the maximal-comparability overlap population. Moving along $\lambda$ trades interpretability of the population for empirical support.
- **$\Gamma$ — the extrapolation axis (how far models may reach).** $\Gamma$ bounds how fast conditional means may vary outside a group's observed support. $\Gamma = 0$ forbids extrapolation (point identification only on common support); $\Gamma = \infty$ yields worst-case Manski bounds. Intermediate $\Gamma$ produce shrinking identified sets for the *original full-population contrast*.
- **$\Lambda$ — the confounding axis (how wrong unconfoundedness may be).** A marginal sensitivity model parameter bounding the odds-distortion of true versus modeled group assignment probabilities. $\Lambda = 1$ is exact unconfoundedness.

The output of a SCOVA analysis is (i) point and simultaneous interval estimates for the contrast process on the identified axis, (ii) bounds with valid interval-coverage on the partially identified axes, and (iii) **certificates**: the largest budget on each axis under which a stated qualitative conclusion (typically the sign of a contrast) survives. A conclusion reported as "$\hat\theta = 4.2$, $p = .01$" becomes "the A–B contrast is positive with simultaneous 95% confidence for all targets with $\lambda \ge 0.3$; its sign survives extrapolation budgets up to $\Gamma^* = 2.1$ (in outcome-SD per covariate-Mahalanobis unit) and confounding up to $\Lambda^* = 1.8$; the full-population contrast is interval-identified in $[0.9, 7.6]$ at $\Gamma = 1$."

This is the conceptual leap beyond both source documents: **the estimand is a process over an epistemic budget space, inference is uniform over that process, and robustness is reported as certified budget thresholds rather than side analyses.** The t-test's single number is recovered as the degenerate corner (randomized assignment, full support, no confounding) — which is exactly where SCOVA should and does collapse to classical efficiency (§12, Proposition P6).

Three further structural commitments carry over from the consolidation:

1. **Comparability structure for $K \ge 2$**: which contrasts are estimable is itself an inferential output, organized as a design-certified graph/hypergraph, replacing the "global F-test + post-hocs" ritual.
2. **Design/analysis firewall**: all target, path, graph, and contrast-family decisions are functions of $(X, A)$ only, locked before outcomes are read; the software enforces this mechanically.
3. **Typed verdicts**: every requested contrast returns one of a small closed set of verdicts (§13) — including refusal — with the evidence for the verdict attached.

## 5. Formal setup and estimands

### 5.1 Data and nuisances

Observe i.i.d. $O_i = (X_i, A_i, Y_i)$, $i = 1, \dots, n$, with pre-membership covariates $X \in \mathcal{X} \subseteq \mathbb{R}^p$, group label $A \in \{1, \dots, K\}$, $K \ge 2$, outcome $Y \in \mathbb{R}$ (v1 scope: continuous or bounded; binary via TMLE backend; survival/count deferred). Nuisances:

$$
e_k(x) = P(A = k \mid X = x), \qquad m_k(x) = E[Y \mid A = k, X = x], \qquad k = 1, \dots, K,
$$

with $e(x) = (e_1(x), \dots, e_K(x))$ on the simplex. Potential-outcome notation $Y(k)$ is used only on the causal rung of the interpretation ladder (§6).

### 5.2 Tilted standardized means

A **target tilt** is a measurable $h: \mathcal{X} \to [0, \infty)$ with $0 < E[h(X)] < \infty$, inducing target covariate law $dQ_h/dP_X \propto h$. The **standardized group mean functional** is

$$
\psi_k(h) \;=\; \frac{E[\,h(X)\, m_k(X)\,]}{E[\,h(X)\,]} \;=\; E_{Q_h}[\,m_k(X)\,],
$$

with contrast $\theta_c(h) = c^\top \psi(h)$ for $c \in \mathbb{R}^K$, $\sum_k c_k = 0$, and contrast family $\mathcal{C}$ (all pairwise; one-vs-rest; ordered trend scores; user-defined; learned — §10.4).

$\psi_k(h)$ is *always well-defined* wherever $m_k$ is identified on $\mathrm{supp}(Q_h)$ — this is the assumption-lean foundation of the interpretation ladder. Its causal reading as $E_{Q_h}[Y(k)]$ requires the additional rungs in §6.

### 5.3 The tilt path (primary estimand process)

Fix a scientific base tilt $h_0$ (default $h_0 \equiv 1$, the study population; alternatives: an external population's density ratio, a policy-defined tilt, or a single group's covariate law for ATT-type standardization). Define the K-way **overlap tilt**

$$
h_{\mathrm{ow}}(x) \;=\; \Big(\sum_{k=1}^{K} \frac{1}{e_k(x)}\Big)^{-1},
$$

the harmonic-mean generalization of the binary overlap weight $e_1 e_2$: bounded by $\min_k e_k(x) \le K \, h_{\mathrm{ow}}(x)$, vanishing wherever any group vanishes, and requiring **no positivity assumption**. Define the **geometric tilt path**

$$
h_\lambda(x) \;=\; h_0(x)^{1-\lambda} \, h_{\mathrm{ow}}(x)^{\lambda}, \qquad \lambda \in [0, 1].
$$

Properties that matter for the theory: (i) $h_\lambda$ is smooth in $e$ for every $\lambda > 0$ with bounded, explicitly computable derivatives $\partial h_\lambda / \partial e_j = \lambda\, h_\lambda \, h_{\mathrm{ow}} / e_j^{2}$; (ii) the map $\lambda \mapsto h_\lambda(x)$ is smooth and monotone in comparability-tilting; (iii) hard trimming $\mathbb{1}\{\min_k e_k \ge \eta\}$ is recovered only as a nonsmooth limit and is deliberately *not* the primitive (Gap 6 resolution). Contrast-specific paths substitute the pairwise overlap tilt $h_{\mathrm{ow}}^{(a,b)}(x) = (e_a^{-1} + e_b^{-1})^{-1}$ for pairs, and subset tilts for cliques of the comparability graph.

The **primary estimand of SCOVA** is the contrast process

$$
\Theta \;=\; \{\, \theta_c(\lambda) := c^\top \psi(h_\lambda) \;:\; c \in \mathcal{C},\; \lambda \in [\lambda_{\min}, 1] \,\},
$$

a random element of $\ell^\infty(\mathcal{C} \times [\lambda_{\min}, 1])$, where $\lambda_{\min} \ge 0$ is design-certified (the smallest tilt at which precision and influence-concentration gates pass; $\lambda_{\min} = 0$ when the scientific target itself is supported). Derived functionals reported with valid inference (via continuous mapping / functional delta method on the uniform limit, §10.1):

- **Sign-certificate set** $\hat\Lambda_+(c) = \{\lambda : \text{simultaneous lower band of } \theta_c(\lambda) > 0\}$ and its infimum $\lambda^\dagger_c$ — "for which populations is the conclusion certified."
- **Target-drift profile** $\lambda \mapsto D(Q_{h_\lambda}, Q_{h_0})$ (e.g., a vector of standardized mean shifts of key covariates plus an energy distance), reported *alongside* the band so the population being certified is always described, answering "overlap populations are scientifically vague" objections with an explicit who-is-in-the-target readout.
- **Stability functional** $\sup_{\lambda} |\theta_c(\lambda) - \theta_c(1)|$ — a certified measure of how target-dependent the conclusion is.

### 5.4 The anchored full-population estimand (partial identification)

The original question — the contrast in the scientific target $\theta_c(h_0)$ — is retained, not abandoned. Decompose, for a smooth support weight $\omega(x) \in [0,1]$ (default: $\omega = h_{\mathrm{ow}} / \max h_{\mathrm{ow}}$ or a smoothed version of $\mathbb{1}\{\min_k e_k \ge \eta_0\}$ with declared $\eta_0$):

$$
\theta_c(h_0) \;=\; \underbrace{E_{Q_{h_0}}[\,\omega(X)\, c^\top m(X)\,]}_{\text{identified component}} \;+\; \underbrace{E_{Q_{h_0}}[\,(1 - \omega(X))\, c^\top m(X)\,]}_{\text{extrapolation component}}.
$$

The second term involves $m_k(x)$ in regions where group $k$ is (nearly) unobserved. SCOVA bounds it under a declared **extrapolation budget**, yielding an identified interval $[\theta_c^{L}(\Gamma), \theta_c^{U}(\Gamma)]$ (construction in §7.2) and the **extrapolation certificate** $\Gamma^*_c = \sup\{\Gamma : \theta_c^{L}(\Gamma) > 0\}$ — an E-value-like quantity for support rather than confounding, which appears to be new.

## 6. The interpretation ladder and tiered assumptions

Rather than a binary causal/descriptive switch, SCOVA defines a ladder; each rung adds assumptions and upgrades the license attached to the *same* estimates. The software requires the analyst to declare a rung; the report language is generated from the declaration.

**Rung 0 — Standardized descriptive contrast (always available).**
- (A0.1) i.i.d. sampling of $(X, A, Y)$ from a common law.
- (A0.2) Estimability: $m_k$ identified on $\mathrm{supp}(Q_{h_\lambda})$ for all groups entering the contrast — guaranteed by construction for $\lambda > 0$ tilts; enforced by gates otherwise.
- License: "$\theta_c(\lambda)$ is the difference in mean outcomes between groups after standardizing both to the common covariate distribution $Q_{h_\lambda}$." No counterfactual language. This is the correct rung for protected attributes and other non-manipulable group labels (disparity/fairness auditing).

**Rung 1 — Transportable association.** Adds (A1.1) no unmeasured effect modification outside $X$ relative to the declared external target (when $h_0$ is external). License: standardized contrasts transport to the named target population.

**Rung 2 — Causal contrast under conditional exchangeability.** Adds:
- (A2.1) Consistency / well-defined group states: $A_i = k \Rightarrow Y_i = Y_i(k)$; group membership admits a coherent intervention or target-trial emulation (declared in prose in the estimand declaration).
- (A2.2) Conditional exchangeability: $\{Y(1), \dots, Y(K)\} \perp A \mid X$.
- (A2.3) Positivity on the target: for the anchored estimand, positivity where $\omega > 0$; for path estimands with $\lambda > 0$, replaced by the bounded-tilt property (no positivity needed — the tilt vanishes where support vanishes).
- (A2.4) No interference (SUTVA); (A2.5) $X$ strictly pre-membership (no mediators/colliders; enforced by a declared causal-role audit of the covariate list, with the framework refusing Rung 2 if any covariate is flagged post-membership).
- License: $\psi_k(h_\lambda) = E_{Q_{h_\lambda}}[Y(k)]$; contrasts are average causal effects in the tilted population.

**Rung 3 — Confounding-robust causal statement.** Adds the marginal sensitivity model (A3.1): the true assignment odds deviate from modeled odds by at most $\Lambda$, i.e. $\Lambda^{-1} \le \dfrac{e_k(x)/(1 - e_k(x))}{e_k^{\dagger}(x, y)/(1 - e_k^{\dagger}(x, y))} \le \Lambda$ where $e_k^\dagger$ conditions additionally on the potential outcome. License: the sign/interval statements hold for all confounding within budget $\Lambda$; the report headlines $\Lambda^*_c$.

Two rules from the source documents are hardened here. First (from HOTC E1.6 / SCGC's pre-registration): **any data-dependence of $h_0$, $\mathcal{C}$, the graph, or gates must be a function of $(X, A)$ only**, with inference interpreted conditionally on that design stage (Theorem T3 makes this precise). Second (from SCGC K2): when (A2.1) is indefensible, the framework does not merely relabel — it *suppresses counterfactual language in generated output* and switches the winner/ranking modules to descriptive phrasing.

---

# Part III — Theory

## 7. Identification and the anchored decomposition

### 7.1 Identification on the path

For any $\lambda$ with $h_\lambda$ bounded and (A0.2), $\psi_k(h_\lambda)$ is identified as $E[h_\lambda(X) m_k(X)] / E[h_\lambda(X)]$ — a functional of the observed-data law. On Rung 2, the same functional equals $E_{Q_{h_\lambda}}[Y(k)]$. Because $h_\lambda \le \text{const} \cdot e_k$ pointwise for every $k$ when $\lambda = 1$ (and the ratio $h_\lambda / e_k$ is controlled for $\lambda > 0$ under a mild tail condition), the identifying functional never divides by small propensities in regions the tilt has not already down-weighted. This is the formal sense in which the path "buys" identification-stability with target drift, and the drift profile (§5.3) prices the purchase.

One honest subtlety absent from both source documents: for $e$-dependent tilts, **the estimand itself depends on the true $e$**. Consistent GPS estimation is therefore required for the *definition* of the target to converge — double robustness protects the outcome-model half, not the target definition. SCOVA states this openly: the GPS model is part of the design, is calibration-audited (§13), and its uncertainty enters the EIF via the correction term of §8.

### 7.2 Bounding the extrapolation component (Gap 3)

Write the extrapolation component for group $k$ as $R_k = E_{Q_{h_0}}[(1-\omega(X))\, m_k(X)]$. SCOVA bounds $m_k(x)$ on the non-support region under one of three nested, declared assumptions:

**(B1) Bounded outcomes (Manski anchor, assumption-minimal).** If $Y \in [L, U]$ (known or declared), then $m_k(x) \in [L, U]$ off-support, giving worst-case bounds — wide but incontestable. This is the rigorous default for bounded scales (test scores, rates, Likert-derived indices).

**(B2) Extrapolation-budget (Lipschitz/Hölder) bounds.** Assume $m_k$ is $\Gamma$-Lipschitz in a declared metric $d$ (default: Mahalanobis on a standardized covariate representation; outcome units per unit distance). Then for $x$ off group $k$'s support $S_k$,

$$
m_k(x) \in \big[\, \tilde m_k(x) - \Gamma\, d(x, S_k),\; \tilde m_k(x) + \Gamma\, d(x, S_k) \,\big],
$$

where $\tilde m_k(x)$ is the regression evaluated at (or averaged over a neighborhood of) the nearest supported point(s) and $d(x, S_k)$ is a support distance estimated by $k$-NN distance to group-$k$ units in the representation space. Integrating against $Q_{h_0}(1-\omega)$ yields $[\theta_c^L(\Gamma), \theta_c^U(\Gamma)]$, continuous and monotone in $\Gamma$, with $\Gamma \to \infty$ recovering (B1) truncated bounds and $\Gamma \to 0$ recovering pure nearest-support imputation (reported only as a bound endpoint, never as a point estimate).

**(B3) Bounded heterogeneity variant.** Assume the *contrast* $m_a - m_b$ (not each surface) is $\Gamma$-Lipschitz — often far more credible (baseline risk varies wildly; incremental group differences vary slowly) and yields much tighter bounds for pairwise contrasts. This is the recommended headline for pairwise anchored reporting.

Inference on the bounds uses the Imbens–Manski/Stoye construction: each endpoint is itself a (mostly) pathwise-differentiable functional with an estimable influence function; the interval CI takes the lower endpoint's lower confidence limit and upper endpoint's upper limit with the IM critical value. The nonsmooth ingredients ($d(x, S_k)$; min/max over neighbors) are handled by (i) fixing the neighborhood rule in the design stage, (ii) a margin condition ruling out mass exactly at the support boundary, and (iii) a conservative smoothing (soft-min) fallback whose validity does not need the margin condition. Proof strategy in §12 (T2).

**The extrapolation certificate ($X$-value).** For a positive estimated contrast, $\Gamma^*_c = \sup\{\Gamma \ge 0 : \theta^L_c(\Gamma) > 0\}$, with a confidence version using the IM band. Interpretation mirrors the E-value: "the conditional mean surfaces would need to change faster than $\Gamma^*$ outcome-SDs per unit covariate distance outside the observed support to overturn the sign." Reporting $\Gamma^*$ next to the confounding certificate $\Lambda^*$ makes the two untestable extrapolations — across support and across unobservables — commensurable objects in one table. To my knowledge no existing group-comparison framework does this.

## 8. Efficient influence functions with estimated tilts (Gap 1)

### 8.1 Fixed tilt (known result, stated for completeness)

For fixed $h$, with $\eta = E[h(X)]$:

$$
\phi_k^{\mathrm{fix}}(O) = \frac{h(X)}{\eta}\Big[ \frac{\mathbb{1}(A=k)}{e_k(X)}\{Y - m_k(X)\} + m_k(X) - \psi_k(h) \Big].
$$

### 8.2 Propensity-dependent tilt: the target correction term

Let $h(x) = g(e(x); \lambda)$ for $g$ continuously differentiable on the simplex interior. The parameter $\psi_k(\lambda) = E[g(e(X);\lambda)\, m_k(X)] / E[g(e(X);\lambda)]$ now depends on the law through three channels: $m_k$, the marginal of $X$, and $e$ *inside the tilt*. Computing the pathwise derivative along regular parametric submodels and collecting terms, the efficient influence function is

$$
\phi_{k,\lambda}(O) \;=\; \frac{1}{\eta_\lambda}\Big[\, h_\lambda(X)\, \frac{\mathbb{1}(A=k)}{e_k(X)}\{Y - m_k(X)\} \;+\; \{m_k(X) - \psi_k(\lambda)\}\Big( h_\lambda(X) + \sum_{j=1}^{K} \frac{\partial g}{\partial e_j}(e(X); \lambda)\,\{\mathbb{1}(A=j) - e_j(X)\} \Big) \Big].
$$

Remarks that belong in the paper:

1. **The correction term $\sum_j \partial_j g \cdot (\mathbb{1}(A=j) - e_j)$ is the new object.** It is invariant to adding constants to the simplex gradient (since $\sum_j(\mathbb{1}(A=j) - e_j) = 0$), so the derivative may be taken in any convenient parametrization. For fixed tilts it vanishes; for $K = 2$, $h = e_1 e_2$ it reproduces the known ATO efficiency structure (Li–Morgan–Zaslavsky's result that estimating the PS *helps* falls out as a corollary — a strong sanity check); for the geometric path, $\partial_j h_\lambda = \lambda h_\lambda h_{\mathrm{ow}} / e_j^2$, bounded because $h_{\mathrm{ow}} \le e_j$.
2. **Neyman orthogonality holds jointly in $(m, e)$**: first-order insensitivity of the moment condition to nuisance perturbations, so cross-fitted plug-in estimation attains $\sqrt n$ inference under the product-rate condition $\|\hat m_k - m_k\|_2 \cdot \|\hat e - e\|_2 = o_P(n^{-1/2})$ plus $\|\hat e - e\|_2 = o_P(n^{-1/4})$ for the quadratic remainder of the tilt channel (the extra requirement created by target estimation — an honest cost, stated, and the reason GPS calibration is a gate rather than a nicety).
3. **Implementation is generic via automatic differentiation.** Because the correction needs only $\partial g / \partial e_j$, implementing tilts as autodiff-traceable functions of $e$ makes the *entire* EIF machinery automatic for any user-declared smooth tilt — a genuine software contribution that keeps the theory and the code in lockstep (§17).
4. The AIPW estimator averages the uncentered EIF over cross-fitting folds; the **TMLE backend** replaces step (iii) with a targeted fluctuation using clever covariate $H_{k,\lambda}(O) = (h_\lambda(X)/\bar h_\lambda)\, \mathbb{1}(A=k)/e_k(X)$ plus a tilt-channel fluctuation, preserving bounds for bounded $Y$.

## 9. Estimation algorithms

**Stage D — Design (uses $(X, A)$ only; $Y$ physically unread).**
1. Declare: rung, base tilt $h_0$, contrast family $\mathcal{C}$, path grid $\Lambda_{\mathrm{grid}}$, gates (min ESS per group per contrast, influence-concentration cap, calibration tolerance, balance tolerance), multiplicity target (FWER level over declared family), extrapolation assumption (B1/B2/B3) and metric, sensitivity model. Serialize and hash the declaration.
2. Cross-fit the GPS $\hat e^{(-v)}$ (calibrated multinomial learners; Super Learner default), $V$ folds.
3. Compute design diagnostics: propensity simplex, $\min_k \hat e_k$ distribution, pairwise/subset design-ESS $\kappa$, balance under candidate tilts.
4. Build the **comparability graph** $\hat G = (\{1..K\}, \hat E)$: edge $(a,b) \in \hat E$ iff pairwise gates pass at some $\lambda \le \lambda_{\max}^{ab}$; hyperedges for subsets passing joint gates. Restrict $\mathcal{C}$ to graph-supported contrasts; record refusals.
5. Lock: $\lambda_{\min}$ per contrast, final $\mathcal{C}$, gates. Emit the design report.

**Stage E — Estimation (outcomes unlocked).**
6. Cross-fit outcome regressions $\hat m_k^{(-v)}$ (group-specific and pooled-with-interactions learners; Super Learner).
7. For each fold, each $\lambda \in \Lambda_{\mathrm{grid}}$, each $k$: evaluate $\hat h_\lambda$, the autodiff correction $\partial g/\partial e$, and the uncentered EIF; assemble the $n \times |\mathcal{C}| \times |\Lambda_{\mathrm{grid}}|$ influence array $\hat\Phi$ (this array is the single computational asset everything downstream reuses).
8. Point estimates $\hat\theta_c(\lambda)$; TMLE variant for bounded outcomes.
9. Anchored module: estimate $\omega$, support distances $d(\cdot, S_k)$, bound endpoints over the declared $\Gamma$ grid, endpoint EIFs.

**Stage I — Inference.** Multiplier bootstrap on $\hat\Phi$ (§10.1); graph-conditional simultaneous intervals (§10.3); winner/ranking modules (§10.4); DR-CRT if triggered (§10.5); sensitivity sweep over $\Lambda$ grid (§11).

**Stage R — Reporting.** Verdict per requested contrast (§13); path bands with target-drift panel; anchored intervals over $\Gamma$; certificate table $(\lambda^\dagger, \Gamma^*, \Lambda^*)$; full diagnostics appendix. Generated language obeys the declared rung.

## 10. Inference engine

### 10.1 Uniform inference on the contrast process

With cross-fitting, smooth tilts, bounded correction derivatives, and the rate conditions of §8.2, $\sqrt n(\hat\theta - \theta)$ converges weakly in $\ell^\infty(\mathcal{C} \times \Lambda_{\mathrm{grid}} \to [\lambda_{\min}, 1])$ to a mean-zero Gaussian process with covariance $E[\phi_{c,\lambda}\phi_{c',\lambda'}]$ (Theorem T1). The index class is a smooth finite-dimensional family (contrasts are a finite set; $\lambda$ enters through uniformly Lipschitz $g_\lambda$), so Donsker manageability is straightforward; the delicate part is uniform control of cross-fitted nuisance remainders, handled by the now-standard fold-wise decomposition plus a uniform-in-$\lambda$ envelope. Simultaneous $(1-\alpha)$ bands come from the **multiplier bootstrap**: draw $\xi_i \sim N(0,1)$, form $\sup_{c, \lambda} |n^{-1/2} \sum_i \xi_i \hat\phi_{i,c,\lambda} / \hat\sigma_{c,\lambda}|$, take its $(1-\alpha)$ quantile. Cost is trivial: one $n \times (|\mathcal{C}|\cdot|\Lambda|)$ matrix, $B$ Gaussian vectors. Derived functionals ($\lambda^\dagger_c$, sign-certificate sets, stability functionals) inherit validity via continuous mapping and, where needed, a functional delta method with directional differentiability handled by the conservative projection construction.

### 10.2 Fixed-family simultaneous testing

Global adjusted null $H_0(\lambda): C\psi(h_\lambda) = 0$ via Wald/$\chi^2$ at declared $\lambda$'s, or sup-Wald along the path with bootstrap critical values. All-pairwise families use the max-$t$ band (Tukey-style, ML-native, DR). FWER is controlled over the **declared** family; FDR variants available for large exploratory families.

### 10.3 Graph-conditional inference (Theorem T3)

Because edges/hyperedges of $\hat G$ are selected by $(X,A)$-measurable rules, and outcome-stage estimation is cross-fitted, simultaneous intervals over the selected family achieve nominal coverage *conditional on the design stage*. The clean version uses an outer design/estimation split; the practical version argues conditional validity from measurability plus cross-fitting. Both are stated; the split version carries the theorem, the practical version carries a simulation-verified corollary. Global "ANOVA-type" claims are only emitted for maximal cliques — the framework structurally cannot assert a K-way conclusion that the design stage did not certify.

### 10.4 Learned contrasts: winner and ranking (from SCGC, hardened)

- **Winner contrast** $\Delta^\star = \max_{j \ne k} \theta_j(\lambda) - \theta_k(\lambda)$: default inference is projection of the max onto the simultaneous band (always valid, mildly conservative); optional sharper mode via bootstrap of the max functional with $\epsilon$-argmax stabilization to handle non-uniqueness (directional Hadamard differentiability); optional conditional-selective mode restricted to structured selections. The report states explicitly which error notion is controlled (simultaneous vs. selective) — SCGC's demand, made mechanical.
- **Ranking**: simultaneous confidence sets for ranks and top-$m$ membership, computed from the joint EIF covariance using inference-on-ranks methodology (Mogstad–Romano–Shaikh–Wilhelm style) applied to standardized tilted means — an apparently novel combination and a directly useful applied deliverable ("league tables with honest uncertainty").
- **Learned contrast vectors / clusterings**: permitted only via discovery/estimation splitting; the software will not emit confirmatory p-values for same-sample selections (typed verdict: `exploratory-only`).

### 10.5 Finite-sample robust backend: the doubly robust conditional randomization test (Gap 5)

When gates are marginal (small ESS, heavy influence tails) the Gaussian approximation is untrustworthy. SCOVA includes a **DR-CRT** for the sharp adjusted null $H_0^{\mathrm{sharp}}$: $Y \perp A \mid X$ (equivalently, on Rung 2, $Y_i(1) = \cdots = Y_i(K)$ plus exchangeability):

1. Hold cross-fitted $\hat e$, $\hat m$ fixed. Compute the observed statistic $T$ = sup over the certified family of studentized AIPW contrasts.
2. For $b = 1..B$: resample labels $A_i^{(b)} \sim \mathrm{Multinomial}(\hat e(X_i))$ independently across $i$; recompute $T^{(b)}$ (weights and indicators change; nuisances fixed).
3. $p = (1 + \#\{T^{(b)} \ge T\}) / (B + 1)$.

Validity: exact if $\hat e = e$ (model-X logic transplanted from conditional-independence testing to multi-group comparison); with estimated $\hat e$, the type-I error excess is bounded by an average total-variation distance between true and fitted conditional label laws (Proposition P5, adapting Berrett-et-al-type robustness bounds), and the DR form of the statistic empirically shrinks sensitivity further. Role: a *robustness backend* that certifies or challenges borderline CLT-based rejections; never a cure for confounding (it tests the adjusted sharp null under the declared rung, nothing more).

## 11. The confounding axis and the robustness surface (Gap 4)

For each certified contrast and each $\lambda$, compute marginal-sensitivity-model bounds $[\theta_c^{L}(\lambda; \Lambda), \theta_c^{U}(\lambda; \Lambda)]$ (sharp MSM bounds via the quantile-balancing formulation, with percentile-bootstrap inference), and the **confounding certificate** $\Lambda^*_c(\lambda) = \sup\{\Lambda : \theta_c^L(\lambda;\Lambda) > 0\}$. Two objects fall out that neither source document has:

1. **The robustness surface** $R_c(\lambda) = \Lambda^*_c(\lambda)$: how robustness to hidden confounding varies as the target tilts toward overlap. Conjecture (to be proven under conditions or established empirically): for bounded outcomes, $R_c(\lambda)$ is typically nondecreasing in $\lambda$, because overlap tilts bound the weights that hidden confounding can exploit. If provable even under restrictions, this is a quotable theorem ("comparability buys robustness"); if not, it is a headline empirical regularity with a counterexample characterization.
2. **The certificate table** $(\lambda^\dagger_c, \Gamma^*_c, \Lambda^*_c)$: one row per contrast, three budgets per row. This is the artifact practitioners will actually cite, and it is the surface realization of the §4 thesis.

## 12. Theorem roadmap (statements to prove, with proof strategies)

**T1 (Uniform estimand-path CLT and bootstrap validity).** Under (A0)+(rung assumptions), smooth tilt family with bounded simplex-gradients, cross-fitting, $\|\hat m_k - m_k\|\cdot\|\hat e - e\| = o_P(n^{-1/2})$, $\|\hat e - e\| = o_P(n^{-1/4})$, and $\inf_\lambda \eta_\lambda > 0$: $\sqrt n(\hat\theta - \theta) \Rightarrow \mathbb{G}$ in $\ell^\infty(\mathcal{C} \times [\lambda_{\min}, 1])$, and the multiplier bootstrap consistently estimates the law of $\sup |\mathbb{G}|$. *Strategy:* fold-wise empirical-process decomposition (DML-style) + second-order remainder expansion of the tilt channel + finite-class-smooth-index Donsker argument + conditional multiplier CLT. Corollaries: validity of $\lambda^\dagger$, sign-certificate sets, stability functionals.

**T2 (Anchored bounds: consistency and interval coverage).** Under (B1) or (B2/B3 with margin or smoothed-distance construction), endpoint estimators are asymptotically linear; Imbens–Manski intervals attain uniform asymptotic coverage of the identified set over the declared $\Gamma$ grid; the certificate $\hat\Gamma^*$ is consistent with a one-sided confidence version. *Strategy:* endpoint EIF derivation treating $d(\cdot, S_k)$ as a fixed design-stage transform under the margin condition; Stoye-type uniformity handling for near-point-identified regimes.

**T3 (Conditional validity over the design-selected graph).** With edge/gate selection measurable w.r.t. $(X, A)$ (or an independent design split) and cross-fitted outcome estimation, simultaneous max-$t$ intervals over the selected contrast family attain nominal coverage conditional on the design stage. *Strategy:* condition on the design $\sigma$-field; the estimation-stage EIFs retain their conditional CLT; the selected family is design-measurable and finite.

**T4 (Winner/selective validity).** (a) Projection method: simultaneous bands imply valid (conservative) inference for any data-dependent selection from the family — immediate. (b) Stabilized bootstrap of the max functional is consistent under $\epsilon$-unique-argmax or via directional-differentiability corrections. *Strategy:* functional delta method for sup-functionals; known non-differentiability caveats stated honestly.

**P5 (DR-CRT level bound).** Type-I error $\le \alpha + c \cdot E[\mathrm{TV}(\mathcal{L}(A \mid X), \hat{\mathcal{L}}(A \mid X))]$ under $H_0^{\mathrm{sharp}}$. *Strategy:* coupling argument following model-X robustness analyses, adapted to multinomial labels and cross-fitting.

**P6 (Classical reductions — the credibility propositions).** (i) Under randomization ($A \perp X$), SCOVA at any $\lambda$ reduces to covariate-adjusted mean comparison and attains the semiparametric efficiency bound; with $h \equiv 1$ and linear working models it is asymptotically equivalent to ANCOVA when ANCOVA's model is correct (no first-order efficiency loss on ANCOVA's home turf — SCGC's K9 demand, answered with a proposition rather than a plea). (ii) At $K = 2$, $\lambda = 1$, the fixed-tilt limit recovers generalized overlap weighting with the known efficiency phenomenon as a special case of the §8.2 correction term. *Strategy:* direct EIF comparison.

---

# Part IV — Practice: Diagnostics, Failure Modes, Incumbents, Simulations, and the Coding Roadmap

## 13. Diagnostics and typed verdicts

All diagnostics are computed objects attached to the output, not optional plots. Consolidated suite (union of the two documents plus additions):

**Design-stage (pre-outcome):** GPS calibration curves and multinomial calibration error (gate); propensity simplex / low-dimensional embeddings; distribution of $\min_k \hat e_k$ under candidate targets; per-contrast design ESS $\kappa_{ab}(\lambda)$ and Kish ESS per group; weighted balance (SMDs on covariates and learned basis functions, energy distance, cross-validated classifier two-sample check); comparability graph with edge weights = certified precision; excluded-mass table ("who is not comparable": size and covariate profile of low-overlap strata).

**Estimation-stage:** influence-concentration index (share of variance from top $r\%$ of $|\hat\phi|$; gate); density-ratio tails $r_k = h_\lambda / e_k$ per group (extrapolation pressure); per-group target-aware extrapolation index (support distance of target mass from each group's observed support — SCGC's H2, formalized as the same $d(\cdot, S_k)$ object the anchored bounds consume, so the diagnostic and the bound share one estimand); learner-sensitivity panel (spread of $\hat\theta_c(\lambda)$ across a declared learner library — model-dependence band); estimator triangulation (OR-only vs IPW vs AIPW vs TMLE); negative-control hooks.

**Typed verdicts (closed set, machine-readable):**
- `certified(λ†, Γ*, Λ*)` — passes gates; certificates attached.
- `certified-overlap-only` — scientific target fails; certified for $\lambda \ge \lambda^\dagger$ with drift profile attached; anchored interval reported for the scientific target.
- `interval-only(Γ)` — point identification refused; anchored bounds only.
- `descriptive-only` — Rung ≥ 2 declaration rejected (manipulability or covariate-role audit failed).
- `exploratory-only` — selection/estimation not separated; no confirmatory inference emitted.
- `refused(reason)` — gates failed everywhere on the path; reasons enumerated (ESS, influence concentration, calibration, balance).

The verdict system is the operational face of "refusal as a first-class result" from both documents, made mechanical so it cannot be quietly overridden.

## 14. Failure modes (explicit, with detection and behavior)

1. **Unmeasured confounding** — undetectable; behavior: mandatory $\Lambda$-sweep, headline $\Lambda^*$, hidden-confounder simulation arm demonstrating honest non-robustness.
2. **Ill-defined group states** (non-manipulable labels) — detection: declaration audit; behavior: Rung 0 language enforced.
3. **Positivity/support failure** — detection: design gates, graph sparsity; behavior: path restriction, anchored bounds, or refusal — never silent extrapolation.
4. **Mediator/collider adjustment** — detection: covariate-role declaration; behavior: refuse Rung 2.
5. **Interference/spillovers** — out of scope v1; declared limitation, refusal trigger for clustered designs unless cluster-robust extension enabled.
6. **Nuisance failure** (miscalibrated GPS, overfit outcome models) — detection: calibration gate, cross-fit stability, triangulation spread; behavior: downgrade or refuse; note the §7.1 caveat that $e$-dependent targets require consistent GPS for estimand definition.
7. **Heavy-tailed influence / tiny ESS** — detection: concentration gate; behavior: DR-CRT backend, or refusal.
8. **Target/contrast shopping** — prevention: hashed declaration, design/analysis firewall enforced in software (outcome column cryptographically unavailable to the design stage), verdicts downgrade any post-hoc additions to `exploratory-only`.
9. **Estimand drift misread as effect** — path conclusions must be read jointly with the drift profile; report template interleaves them.
10. **Rare groups** — pairwise graph typically retains some edges; K-way claims refused; rare-group scenario in the simulation suite checks for false precision.

## 15. Confrontation with incumbent methods

| Incumbent | What it solves | Why insufficient here | SCOVA relation |
|---|---|---|---|
| t-test / ANOVA | Marginal mean comparison under exchangeability | No standardization, no identification, no support awareness | Degenerate corner (P6); recovered under randomization |
| ANCOVA | Linear adjustment | Implicit target, hidden extrapolation, model-correctness dependence | Special-case nuisance; matched efficiency on its home turf (P6i); beaten under nonlinearity/interaction |
| Matching | Transparent local design | Awkward for $K>2$ and high dims; estimand shifts implicitly | Design-stage comparator/diagnostic |
| GPS / IPW | Multi-valued adjustment | Unstable under weak overlap; positivity-dependent | Nuisance input; path tilts bound the weights |
| Overlap weighting | Bounded weights, ATO target | Single fixed target; no path, no multi-contrast calculus, no anchoring | $\lambda = 1$ endpoint; §8.2 generalizes its EIF structure |
| AIPW / DR, DML | Orthogonal ML inference | Fixed known target; one contrast at a time; no support structure | Estimation engine (declared non-novel) |
| TMLE | Targeted, bounded plug-in | Same target/overlap dependence | Backend for bounded outcomes |
| Causal forests | CATE heterogeneity | ITE focus; aggregation target secondary | Nuisance/heterogeneity diagnostics only; no ITE claims |
| BART / Bayesian hierarchical | Flexible surfaces + uncertainty | Posterior can hide support extrapolation; prior-sensitive | Nuisance learner + comparator; support diagnostics imposed on top |
| Permutation tests | Exact under exchangeability | Labels not exchangeable observationally | Replaced by DR-CRT with explicit level bound (P5) |
| Conformal methods | Finite-sample predictive intervals | Prediction ≠ mean-contrast inference | Optional per-unit counterfactual-uncertainty diagnostic only |
| Manski bounds / partial ID | Assumption-free honesty | Usually detached from ML estimation pipelines and multi-group practice | Absorbed as the $\Gamma$-axis anchor (B1–B3) |
| E-values / MSM sensitivity | Confounding robustness quantification | Single-axis; not target- or support-aware | Absorbed as the $\Lambda$-axis; extended to the robustness surface $R_c(\lambda)$ |

Novelty claim, stated in the ruthless style SCGC demands: no new identification result; the estimation engine is standard. The contributions are (1) the estimand *process* with uniform inference including the estimated-tilt EIF correction (T1, §8.2); (2) the anchored partial-identification layer and extrapolation certificate connecting overlap-restricted answers back to the original question (T2, §7.2); (3) graph-conditional simultaneous inference for multi-group comparability (T3); (4) the three-axis certificate system unifying target, support, and confounding robustness (§11); (5) the DR-CRT finite-sample backend (P5); plus the winner/ranking inference packaging (T4). Items 1–2 are theorem-bearing and, to the best of current knowledge, not available in existing literature in this form; items 3–5 are strong integrations with provable properties. That is a defensible top-tier portfolio without overclaiming.

## 16. Simulation program (merged and pre-registered)

**Factors:** $K \in \{2, 4, 8\}$; $n \in \{500, 2000, 10000\}$; $p \in \{5, 20, 100\}$; group imbalance (balanced / one rare group); response surfaces (linear no-interaction [ANCOVA-favorable], nonlinear + group×covariate interaction, threshold/piecewise); overlap regimes (strong / moderate / near-violation / *pairwise-without-K-way* [HOTC Scenario 5]); confounding strength; hidden-confounder arm; heteroskedastic and heavy-tailed errors; continuous and bounded outcomes.

**Competitors, fairly tuned:** unadjusted ANOVA; ANCOVA; flexible OR-only g-computation; GPS matching; IPW; trimming+IPW; generalized overlap weighting; CBPS/entropy balancing; parametric AIPW; cross-fitted DML; TMLE+SL; causal forest aggregated to targets; BART/BCF; Bayesian hierarchical model; naive max-contrast selection; residual permutation.

**Metrics:** bias/RMSE per estimand *matched to each method's own target* (HOTC's fairness rule); per-contrast and simultaneous coverage; uniform band coverage along the path; selective type-I for the winner under the global null (SCGC's headline; naive max must fail, SCOVA must hold); FWER over graph-selected families; power at fixed FWER; anchored-interval coverage of the true full-population contrast across $\Gamma$; certificate accuracy ($\hat\Gamma^*, \hat\Lambda^*$ vs. oracle flip thresholds); DR-CRT level under weak overlap where Wald fails; diagnostic ROC (do gates flag exactly the unreliable regimes); refusal calibration (refusals concentrate where all methods are bad); runtime.

**Pre-registered win conditions:** (a) efficiency loss $\le$ small vs. ANCOVA/BART on their home turf; (b) correct simultaneous+selective coverage where incumbents fail; (c) anchored intervals cover the full-population truth at declared $\Gamma$ when the truth satisfies the budget, and honestly fail when it does not; (d) under hidden confounding, point estimates are biased *and* $\Lambda^*$ correctly reports fragility; (e) the comparability graph identifies pairwise-estimable structure and refuses unsupported K-way claims; (f) DR-CRT holds level in regimes where the CLT-based test is anti-conservative. Failure on (c), (d), or (f) falsifies the framework's central promises — the suite is designed to be able to kill it.

## 17. Software architecture and coding roadmap

**Delivery strategy.** Build a Python package and import namespace named `scova`. The first releasable product is a trustworthy fixed-target, multi-group analysis core; research-heavy extensions remain isolated and explicitly experimental until their theorem-specific validation gates pass. Stages 0–2 therefore define the core release boundary, and later stages must not delay that release. JAX supplies automatic differentiation for tilt gradients, nuisance learners follow scikit-learn protocols, and inference downstream of the influence array $\hat\Phi$ remains pure linear algebra.

### 17.1 Public interfaces and package boundaries

- `SCOVADeclaration` records the interpretation rung, dataset roles, targets, contrast family, diagnostic gates, random seed, and $\Gamma$/$\Lambda$ grids. Locking it produces a stable hash and prevents outcome-informed design changes.
- `SCOVA` is the cross-fitted analysis object and primary entry point. Its fitted interface exposes `fit(...)`, `contrast(...)`, `infer(...)`, `diagnostics()`, and `report()`.
- `SCOVAResult` stores estimates, influence values, intervals, diagnostics, certificates, verdicts, fold assignments, learner metadata, declaration hash, schema version, and package version.
- `Verdict` is a closed tagged union covering `certified`, `certified-overlap-only`, `interval-only`, `descriptive-only`, `exploratory-only`, and `refused`; every non-certified variant carries machine-readable reasons.
- Nuisance estimators implement scikit-learn-compatible `fit`, `predict`, and, where appropriate, `predict_proba` methods. Result serialization is versioned and round-trippable; incompatible schema changes require an explicit migration.

The internal package is organized around `declaration`, `nuisance`, `estimator`, `inference`, `tilts`, `diagnostics`, `graph`, `anchor`, `sensitivity`, `report`, and `simulate`. Optional learners and experimental inference backends must not be imported by the fixed-target core.

### 17.2 Staged implementation

**Stage 0 — Specification and validation foundation.** Dependencies: none. Deliverables: establish `src/scova`, tests, documentation, CI, typed configuration, deterministic seed handling, and versioned result serialization; define declaration, dataset-schema, contrast, diagnostic, result, and verdict models; build reusable §16 DGPs and the numerical EIF perturbation harness before estimators depend on them. Acceptance gate: the package installs in a clean environment, schemas round-trip without loss, fixed seeds reproduce folds and simulations, and CI runs unit, typing, serialization, and simulation smoke tests. Theory/simulation link: encode the randomized, nonlinear, weak-overlap, and pairwise-without-K-way oracle DGPs needed by P6 and T1–T4.

**Stage 1 — Fixed-target MVP.** Dependencies: Stage 0. Deliverables: implement deterministic cross-fitting, multinomial propensity estimation, outcome regression, fixed-target multi-group AIPW, predefined and user-supplied contrasts, influence values, standard errors, and pointwise confidence intervals for continuous outcomes; accept pluggable scikit-learn-compatible nuisance learners; add propensity calibration, covariate balance, Kish ESS, and influence-concentration diagnostics. Acceptance gate: estimates recover known AIPW special cases, agree with analytic randomized-data results, remain invariant to group-label ordering, and attain nominal pointwise coverage in baseline simulations. Theory/simulation link: validate the fixed-tilt EIF (§8.1), double-robust remainder behavior, and P6 randomized-assignment reduction.

**Stage 2 — Simultaneous multi-group inference.** Dependencies: Stage 1. Deliverables: persist the influence array; implement multiplier-bootstrap max-$t$ inference, simultaneous contrast intervals, global tests, deterministic bootstrap streams, machine-readable verdicts, and the public `SCOVA`/`SCOVAResult` fitted API. Acceptance gate: simulations demonstrate controlled family-wise error over declared finite contrast families; repeated runs with the same seed reproduce intervals and verdicts; serialization preserves all quantities required to rerun inference without refitting nuisances. Theory/simulation link: validate the fixed-family portion of T1 and the projection basis for T4. Completion of this stage marks the fixed-target core as releaseable.

**Stage 3 — Smooth overlap paths.** Dependencies: Stage 2 and a stable estimated-tilt EIF derivation. Deliverables: implement JAX-traceable base, K-way, pairwise, and subset tilts; propensity-dependent tilt corrections; path evaluation; uniform bands; drift profiles; and sign/stability certificates. Add simplex-gradient invariance, automatic-versus-analytic gradient comparisons, and finite-difference EIF perturbation tests. Acceptance gate: fixed-path endpoints reproduce Stage 1 and published overlap-weighting special cases, gradient checks pass declared tolerances, and uniform-band coverage meets preregistered targets across the path grid. Theory/simulation link: T1 and P6(ii).

**Stage 4 — Comparability graph and design firewall.** Dependencies: Stage 3. Deliverables: implement outcome-blind declaration locking, design-stage gates, pairwise/subset comparability scores, graph construction, clique discovery, graph-conditional contrast families, and conditional inference; changes to a locked target or contrast family produce `exploratory-only` results rather than silently replacing the declaration. Acceptance gate: attempts to access outcomes during design fail structurally; pairwise-without-K-way simulations retain supported edges while refusing unsupported K-way claims; graph-selected families retain nominal conditional family-wise coverage. Theory/simulation link: T3 and §16's structural-overlap scenarios.

**Stage 5 — Anchored partial identification.** Dependencies: Stage 4 and finalized B1–B3 endpoint definitions. Deliverables: add standardized support-distance estimation, bounded-outcome and Lipschitz/Hölder extrapolation models, identified-set endpoints, endpoint influence functions, Imbens–Manski intervals, and $\Gamma^*$ certificates. Acceptance gate: endpoint and perturbation tests pass; intervals cover the full-population truth whenever the simulated DGP satisfies the declared budget; intervals widen monotonically with weaker assumptions and downgrade or refuse when required inputs or support conditions fail. Theory/simulation link: T2 and the §7.2 anchored decomposition.

**Stage 6 — Sensitivity and advanced inference.** Dependencies: Stages 3–5, with each submodule independently feature-gated. Deliverables: implement marginal-sensitivity bounds, $\Lambda^*$, robustness surfaces, winner and ranking inference, TMLE for bounded outcomes, and the DR-CRT fallback. Mark each backend experimental until its own gate passes; failure of one backend must not invalidate core AIPW results. Acceptance gate: winner/ranking procedures control the declared selective or simultaneous error rate under the global null; sensitivity bounds reproduce oracle or benchmark cases; TMLE respects outcome bounds; DR-CRT calibration meets preregistered tolerances in weak-overlap scenarios where Wald inference degrades. Theory/simulation link: T4, P5, and the robustness-surface investigation in §11.

**Stage 7 — Release hardening.** Dependencies: stable outputs from all modules selected for the release. Deliverables: run the full §16 factorial benchmark; add one $K \ge 4$ comparative-effectiveness case study and one Rung-0 disparities audit; stabilize reports, plotting, serialization, CLI behavior, and API documentation; publish a compatibility policy, release checklist, and benchmark artifacts. Acceptance gate: runtime and coverage benchmarks are documented and reproducible, examples run from clean environments, public APIs pass backward-compatibility checks, and no correctness-critical issue remains unresolved. Theory/simulation link: consolidate evidence for T1–T4 and P5–P6 and clearly label any feature whose proof or validation remains incomplete.

### 17.3 Validation policy

The numerical harness is non-negotiable: (i) EIF perturbation tests compare $d\psi/d\epsilon$ along constructed submodels with $E[\phi \cdot s]$; (ii) reduction tests recover published AIPW/ATO limits and ANCOVA-comparable efficiency under randomization; (iii) coverage smoke tests run on small grids in CI, while the full factorial suite runs on scheduled or release workflows; and (iv) every stochastic API accepts an explicit seed and records it in `SCOVAResult`. A stage is complete only when its acceptance gate is automated or accompanied by a versioned benchmark artifact.

## 18. Positioning, paper plan, and honest limitations

**Positioning sentence:** *SCOVA is a framework for honest comparison of two or more naturally occurring groups that treats the target population, the common-support structure, the extrapolation budget, and the confounding budget as declared, inferentially-managed components of the estimand — delivering uniform inference over an estimand path, partial-identification anchoring to the original full-population question, graph-structured multi-group comparability, and certified robustness on three axes.* It is explicitly not a successor to the t-test; it reduces to classical procedures exactly where classical assumptions hold.

**Paper 1 (theory + method, top-tier methods venue):** estimand process + estimated-tilt EIF (T1, §8.2); anchored decomposition + $\Gamma^*$ (T2); graph-conditional inference (T3); P6 reductions; core simulations; one application. **Paper 2 (inference extensions):** winner/ranking selective inference (T4), DR-CRT (P5), robustness surface. **Paper 3 (software + applied tutorial, JSS/similar).** This split keeps each paper theorem-anchored, answering the "workflow, not a method" objection both documents flagged as the biggest rejection risk.

**Standing limitations, stated without hedging:** no observed-data method repairs unmeasured confounding — SCOVA prices it ($\Lambda^*$), nothing more; causal readings require manipulability the framework audits but cannot manufacture; $e$-dependent targets make the estimand GPS-dependent (§7.1); interference, longitudinal regimes, and survival outcomes are out of scope for v1; extrapolation budgets (B2/B3) are assumptions, and the framework's honesty consists in indexing conclusions by them rather than pretending they are testable.
