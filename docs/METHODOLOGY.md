# Methodology and statistical judgement

## Target and population

Independent borrowers are observed annually on 1 January. A default must occur strictly after observation and on or before the date twelve calendar months later. Both default and non-default outcomes require a mature horizon at the declared as-of date. Borrower IDs cannot recur; repeated borrower longitudinal modelling is outside this version.

## Synthetic mechanism

Segments prime/standard/vulnerable have normal weights 0.62/0.25/0.13 and log-odds offsets 0/0.5/1.1. A normal latent borrower factor correlates lognormal income, logistic DTI, utilisation and Poisson delinquency. Loan amount scales with income and DTI. Annual macro covariates follow a deterministic sinusoidal cycle. Conditional on these covariates, outcomes are independent Bernoulli draws.

The generating logit is:

```text
intercept - 0.45 log(income / 50000)
 + 2.1 (DTI - 0.3) + 1.4 (utilisation - 0.35)
 + 0.55 delinquency + 0.2 log(loan / 25000)
 + 0.24 (unemployment - 5.5) - 0.16 (GDP growth - 1.6)
 + segment offset + 1.8 max(utilisation - 0.65, 0)^2
 + 0.8 DTI * 1(delinquency > 0)
```

Missingness affects four numerical fields with configurable probability, multiplied by 1.3 for borrowers with previous delinquency. This occurs after true PD generation. Oracle probabilities remain separate from model features. These coefficient values and thresholds are illustrative, not estimates from a bank.

Downturn adds 4.5 percentage points unemployment, subtracts 4.0 percentage points GDP growth, and increases DTI/utilisation by 0.12/0.18. Composition shift changes segment weights to 0.15/0.35/0.50. Mechanism shift adds 1.7 to log-odds, another 1.1 for vulnerable borrowers and 1.5 when utilisation exceeds 0.6. Stress acts only in the configured final test period. Calibration-error injects a -1.5 logit shift into reported final predictions; it demonstrates a prediction pipeline defect, not a refitted model.

## Development protocol

Training fits imputers, missing indicators, scaler, categories and classifiers. Candidate hyperparameters are selected by tuning log loss with deterministic tie resolution. The selected train-only model is retained without train-plus-tune refitting. Calibration fits on a later disjoint mature sample. All predictions on the final test window are made with frozen model and calibration objects.

Default windows are train 2013–2017, tune 2019, calibration 2021 and test 2023–2024. The intervening years are unused. Every previous stage's latest label maturity must strictly precede the following stage's earliest observation. This is intentionally conservative and auditable.

Sigmoid calibration fits intercept and non-negative slope against predicted logits using maximum likelihood; the non-negative slope preserves ordering. Isotonic calibration is flexible but may create ties and overfit with sparse defaults. Identity calibration is supported for comparisons. Probability values exactly zero/one are clipped only where logit calculations require finite scores.

## Metrics, uncertainty and conclusions

AUC measures ranking, Gini is `2*AUC-1`, Brier and log loss measure probability prediction quality. Single-class populations have undefined AUC/Gini represented by JSON null. Diagnostic calibration intercept/slope are estimated on test outcomes solely to describe final performance; no fitted test mapping is applied.

Observed-rate intervals use Wilson's binomial interval. Ordinary borrower bootstrap intervals cover AUC/Gini, Brier, log loss and observed-minus-predicted gap. Resamples with no defaults or no non-defaults are skipped only for AUC/Gini. Paired challenger comparisons use identical resampled borrowers; positive AUC deltas and negative loss deltas favour the challenger. Null calibration intervals use `z*sqrt(sum(p*(1-p)))/n`, conditional on independent Bernoulli defaults. An alarm requires both a gap outside this interval and configured absolute materiality. This approximation can be inaccurate with very sparse expected defaults.

These intervals are conditional on a fixed macro path and fitted model and do not quantify refitting uncertainty or real-world default dependence. Default counts below the configured minimum trigger a limitation. Period and segment results are exploratory without multiplicity adjustment. No universal claim of statistical significance or automatic model approval is made.

PSI bins come from training quantiles and include missing values plus 0.5-count smoothing. Constant training variables have explicit out-of-range bins. Categorical bins include unseen current levels. PSI thresholds are descriptive review triggers, not calibrated statistical tests.

Logistic coefficients are checked against mechanism directions and refitted separately for training years to disclose stability. Train-only feature perturbations assess prediction sensitivity, not causality. Correlated macro variables, different standardisation across annual windows and low default counts can produce unstable signs. No coefficient confidence interval is claimed. The oracle benchmark is evaluated only after all training and calibration are finished.

Findings identify affected model, prediction variant and population, and provide actions. Calibrated-prediction material bias leads to a restriction recommendation. Other concerns lead to further review. Raw scores are reported for comparison. Final test outcomes never select a deployable champion. Remediation requires another mature cohort.

## Limits and references

No real borrower data, production governance, censoring/prepayment, repeated borrowers, contagion, losses, long-run PD or regulatory capital calculation is represented. The toolkit has no relationship or endorsement from Danske Bank. It is designed to support a discussion of model validation judgement.

- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): disjoint calibration data, sigmoid/isotonic trade-offs.
- [BCBS Studies on the Validation of Internal Rating Systems (2005)](https://www.bis.org/publ/bcbs_wp14.pdf): discrimination, calibration and validation limitations.
- [BCBS low-default portfolios (2005)](https://www.bis.org/publ/bcbs_nl6.pdf): limited information from scarce default observations.
