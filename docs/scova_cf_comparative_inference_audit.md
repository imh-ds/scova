# SCOVA-CF comparative-study inference audit

This audit applies to the frozen v2 two-group comparative-methods study. It corrects
the interpretation of interval coverage; it does not alter the frozen artifact,
estimators, DGPs, or point-estimation summaries.

The complete final artifact is from GitHub Actions run
[31575961864](https://github.com/imh-ds/scova/actions/runs/31575961864), with checksum
`fe7fa979847961fea2756c98ce697cca84be2a108aa62a8de1be34060a603ef5`.

| Method | Point-estimation result | Interval basis | Coverage interpretation |
| --- | --- | --- | --- |
| SCOVA-CF | Eligible-population ATE | Cross-fitted AIPW influence-function SE | Reported as a simulation coverage result. |
| Independent AIPW | Eligible-population ATE | Cross-fitted score empirical-SE | Reported as a simulation coverage result. |
| Interacting linear ANCOVA | Eligible-population ATE | Model-based OLS covariance | Reported as model-dependent simulation coverage. |
| PSM | Matched-treated ATT | SD of paired differences | Not evaluated: it omits uncertainty from estimating the propensity score and matching. |
| EconML DRLearner | Eligible-population ATE | SD of predicted individual effects | Not evaluated: this is CATE dispersion, not the sampling SE of the mean ATE. |
| Conservative EconML DRLearner | Eligible-population ATE | SD of predicted individual effects | Not evaluated for the same reason. |

The v2 artifact retains its raw `standard_error` and calculated coverage fields for
reproducibility. The report renderer suppresses coverage for the three
not-evaluated interval bases so they cannot be read as an inferential ranking.

Future comparative protocols must either provide a valid sampling-uncertainty method
for an estimator's declared ATE/ATT, or restrict that estimator to point-estimation
metrics such as bias, RMSE, and tail error. Any corrected DRLearner inferential
implementation requires a new protocol version and a separate evidence set; it must
not be pooled with v2.
