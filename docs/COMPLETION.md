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
| Git workflow | Local development branch prepared; supplied remote is `https://github.com/kkkkkkemi/Quant-risk.git` |
| GitHub Actions | Workflow implemented; current hosted verification is tracked in the development PR checks |

## Substantive baseline result

The baseline final cohort has 3,600 borrowers and 196 defaults (5.44%). Logistic regression and gradient boosting both have AUC approximately 0.77. Raw mean PDs are approximately 5.62% and 5.36%; independent sigmoid calibration reduces them to approximately 4.42% and 4.06%. The configured validation therefore identifies material calibrated-PD underestimation rather than presenting the baseline as approved.

This illustrates calibration sampling/transfer risk: calibrating on an independent later window does not guarantee improved final-period calibration. Development data or test thresholds were not changed to manufacture a passing result. The reports recommend fresh mature data and reviewed recalibration before model use.

## GitHub completion boundary

At the initial local verification, repository access required authentication. The user subsequently signed in, and the repository was confirmed to be an empty private repository. The project is being imported through the authenticated browser on `feature/pd-validation-toolkit`, preserving a separate `main` base for a pull request. Local Git credentials remain independent of browser sign-in. No remote history is overwritten. This local record predates the first hosted run: review the development PR checks and Actions artifacts for the current online execution result rather than treating local success as hosted success.
