# Delivery and verification record

Local verification completed on 2026-09-27 using Python 3.12.14 on Windows.

| Deliverable | Status and evidence |
|---|---|
| Installable package | Wheel and source distribution built under `dist/` |
| Clean installation | Fresh virtual environment, locked dependencies, installed wheel, full quick experiment and manifest verification passed |
| Automated tests | 49 passed; 94% measured package statement coverage |
| Code checks | Ruff lint and format checks passed |
| Scenario demonstration | Six complete modelling scenarios and four correctly blocked integrity defects |
| Failure identification | Calibration bias in downturn, mechanism shift and injected calibration-error cases; population shifts in composition/downturn |
| PDF report | Seven-page baseline report rendered and visually inspected |
| Evidence integrity | All six scenario manifests verified successfully; automated tamper test passed |
| Repeatability | Independent-run predictions and validation results match in the end-to-end test |
| Git workflow | Complete project published on `feature/pd-validation-toolkit`; [pull request #1](https://github.com/kkkkkkemi/Quant-risk/pull/1) opened for review |
| GitHub Actions | [Hosted run](https://github.com/kkkkkkemi/Quant-risk/actions/runs/36329160142) succeeded: Python 3.11/3.12 checks and the full scenario demonstration; reports and evidence retained as artifacts |
| Language | Project documentation, source, configuration, tests, command-line output and generated reports use English |

## Substantive baseline result

The baseline final cohort has 3,600 borrowers and 196 defaults (5.44%). Logistic regression and gradient boosting both have AUC approximately 0.77. Raw mean PDs are approximately 5.62% and 5.36%; independent sigmoid calibration reduces them to approximately 4.42% and 4.06%. The configured validation therefore identifies material calibrated-PD underestimation rather than presenting the baseline as approved.

This illustrates calibration sampling/transfer risk: calibrating on an independent later window does not guarantee improved final-period calibration. Development data or test thresholds were not changed to manufacture a passing result. The reports recommend fresh mature data and reviewed recalibration before model use.

## Hosted verification and review

The supplied repository was confirmed to be an empty private repository after authentication. The complete project was published on `feature/pd-validation-toolkit` with a separate `main` base and an open pull request. No remote history was overwritten. Hosted workflow run `36329160142`, for revision `e9e8837272fcecafb1114bec6398b26a363c2aae`, completed successfully and produced package/coverage artifacts for both Python versions plus the full PDF/evidence artifact. Later documentation updates are checked by subsequent runs; the pull request shows the current revision's status. The pull request remains open for review.
