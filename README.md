# One-Year PD Model Development and Validation Toolkit

An installable, configurable and testable Python toolkit for one-year probability of default (PD) development and validation. It uses **synthetic data from independent borrowers** to demonstrate credit risk modelling, statistical judgement, model validation and reproducible engineering. It does not claim banking production readiness or regulatory compliance.

## Quick start

Requires Python 3.11 or later. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pd-toolkit.exe run --config configs/demo.yaml --output output/demo
.\.venv\Scripts\pd-toolkit.exe verify output/demo
```

On macOS or Linux:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pd-toolkit run --config configs/demo.yaml --output output/demo
pd-toolkit verify output/demo
```

Use `configs/quick.yaml` for a faster demonstration with wider statistical uncertainty. The output directory must be empty; use a new directory for each experiment to preserve its evidence.

Run all scenarios:

```sh
pd-toolkit suite --config configs/demo.yaml --output output/scenarios
python scripts/check_demo.py output/scenarios
```

To change models or calibration, copy a configuration and edit `models.names` (`logistic`, `gradient_boosting`) or `models.calibration` (`sigmoid`, `isotonic`, `none`). Omitted fields receive defaults; unknown configuration keys raise an error. Generate a complete configuration with `pd-toolkit init-config my-config.yaml`.

## Implemented workflow

Generate synthetic data → Check integrity → Train and tune → Calibrate independently → Validate across time and segments → Compare models → Produce a PDF report and evidence bundle.

- Baseline: logistic regression with annual training-period coefficient stability and expected-sign checks.
- Challenger: histogram gradient boosting with a fixed random seed and implicit random early stopping disabled.
- Chronological separation: training in 2013–2017, tuning in 2019, calibration in 2021 and final testing in 2023–2024. Intervening years allow twelve-month outcomes to mature before subsequent stages.
- Preprocessing: medians, missingness indicators, scaling and categorical encoding fitted only on the training sample.
- Validation: AUC, Gini, Brier score, log loss, calibration slope/intercept, calibration curves, Wilson intervals for observed default rates, bootstrap intervals, paired model comparisons, annual/segment evaluation and PSI.
- Conceptual review: comparison with the synthetic mechanism, coefficient stability, training-sample local sensitivity and a separately retained oracle benchmark.
- Reproducibility: random seeds, configuration, dependency versions, Git state, source/data/artifact SHA-256 hashes, model parameters and serialized models.

## Scenarios and deliberate defects

| Scenario | Design | Expected outcome |
|---|---|---|
| `baseline` | Low default rates, correlated features, nonlinear effects and missing values | Establish a reference case while reporting statistical limitations |
| `downturn` | Higher test-period unemployment, lower GDP growth, increased debt and utilisation | Assess stress performance and possible recalibration or use restrictions |
| `composition_shift` | Larger share of higher-risk borrowers in the test period | Identify portfolio changes through PSI and segment evaluation |
| `mechanism_shift` | Changed test-period default-generating mechanism | Identify material PD underestimation |
| `calibration_error` | Shift test prediction logits down by 1.5 | Identify calibration failure while broadly preserving ranking |
| `omitted_variable` | Exclude previous delinquency from modelling | Disclose misspecification and assess its consequences |
| `leakage` | Introduce a post-default collections variable | Block fitting and retain integrity-failure evidence |
| `immature` | Include outcomes that have not matured | Block fitting |
| `duplicate` | Duplicate borrower records | Block fitting |
| `unavailable_feature` | Make income available only after the observation date | Block fitting |

Stress changes affect only the final test period; development data remain identical across scenarios with the same random seed. The `suite` command treats detection of predefined integrity defects as the expected outcome. The ordinary `run` command exits with code 2 for these defects.

## Outputs and review

Each successful experiment generates `report.pdf` and `evidence.zip`, together with inspectable supporting files:

| File | Contents |
|---|---|
| `predictions.csv` | Raw/calibrated final-test predictions, model, year, segment and realised defaults |
| `metrics.csv` / `validation.json` | Overall, annual and segment metrics, intervals, limitations and conclusions |
| `findings.csv` | Severity, affected populations, findings and recommended actions |
| `calibration_bins.csv` / `figures/` | Calibration bins, confidence intervals and plots |
| `tuning.json` / `comparisons.json` | Independent tuning results and paired model comparisons |
| `coefficients.csv` / `sensitivities.csv` | Coefficient stability and local feature sensitivity |
| `observations.csv` | Observable features, timestamps and realised outcomes |
| `oracle.csv` | Separately retained underlying synthetic PDs, unavailable to model fitting |
| `feature_availability.csv` / `integrity.json` | Feature availability times and integrity checks |
| `models/` | Fitted models and calibrators; load only joblib files from trusted sources |
| `config.yaml` / `metadata.json` | Configuration, environment, versions, Git state, parameters and hashes |
| `manifest.json` | Per-file SHA-256 checks, excluding the manifest itself and the ZIP |

Blocked scenarios save `integrity_failure.json` and do not produce a report implying successful validation.

## Tests and packaging

```sh
ruff check .
ruff format --check .
pytest --cov=pdtoolkit --cov-report=term-missing
python -m build
```

Tests cover known metric values, samples with no defaults, invalid probabilities, outcome maturity, preprocessing isolation, chronological boundaries, deliberate defects, final-label changes leaving fitted models unchanged, report generation, model serialization, hash tampering and repeatability.

For the pinned reproduction environment, install `requirements-lock.txt` before the package:

```sh
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
pd-toolkit run --config configs/demo.yaml --output output/repeat
python scripts/reproduce.py output/demo output/repeat
```

With the same source, configuration, dependencies and platform, predictions and metrics use an absolute comparison tolerance of `1e-10`. Execution times, timestamps and PDF metadata need not match byte for byte. Model fitting and key predictions use single-threaded numerical operations.

## GitHub delivery

The [development pull request](https://github.com/kkkkkkemi/Quant-risk/pull/1) contains the complete project. A [hosted workflow run](https://github.com/kkkkkkemi/Quant-risk/actions/runs/36329160142) successfully completed both Python test jobs and the full scenario demonstration on 2026-09-27. Review the PR checks for the latest revision.

`.github/workflows/ci.yml` checks formatting, runs tests and builds packages on Python 3.11/3.12. A dependent job runs the complete demonstration with pinned dependencies on Python 3.12, checks detection of predefined failures and retains PDF reports and evidence for 30 days. The workflow has read-only repository permissions. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Project structure

```text
src/pdtoolkit/   config, data, checks, preprocessing, models, calibration,
                metrics, validation, reporting, pipeline, cli
configs/        Full demonstration and quick configurations
tests/          Data integrity, metrics, workflow and deliberate-defect tests
scripts/        Demonstration acceptance and reproduction comparisons
docs/           Methodology, architecture and completion record
.github/        Actions workflow and pull request template
output/         Generated local reports and evidence (not committed)
```

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for statistical methods and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for engineering design. The initial scope covers one-year PD; LGD, EAD/conversion factors and IFRS 9 remain future extensions.
