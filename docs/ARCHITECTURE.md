# Engineering design

The `src` layout prevents tests accidentally importing unrelated working-directory code. A console entry point and `python -m pdtoolkit` share one interface. Dataclass configuration rejects unknown YAML fields. All stages are explicit modules:

| Module | Responsibility |
|---|---|
| config | Validate configuration and chronological design |
| data | Independent yearly RNG streams, observable snapshots, oracle and availability |
| checks | Fail closed on timing, identifiers, feature allowlist and target defects |
| preprocessing | Training-fitted imputation, missing indicators, scaling and category encoding |
| models | Train/tune candidate search, frozen pipeline, coefficient stability |
| calibration | Monotone sigmoid, isotonic or identity |
| metrics | Explicit bounds, formulas, uncertainty and paired comparisons |
| validation | Overall/period/segment metrics, drift, findings and sensitivities |
| reporting | Deterministic content, charts and paginated PDF |
| pipeline | Orchestration, model/evidence export, fingerprints and manifest |
| cli | Command parsing, scenario suite and explicit exit codes |

Only train/tune/calibrate frames are read inside model development. Oracle information is consulted only after fitting. An output directory must be empty to prevent overwriting earlier evidence. Invalid input produces a structured error artifact. Successful experiments create a manifest before ZIP packaging, excluding self-referential hashes. Manifest verification guards against paths outside the evidence root; it checks hashed files, not authenticity of an untrusted manifest.

`joblib` is used for local trusted model persistence; arbitrary untrusted serialized models must not be loaded. There is no serving endpoint, external data connector or credential storage.

CI uses minimum permissions, runs a Python version matrix, builds distributions and executes a locked demonstration. Lock files capture the actual verified dependency versions. The tests use a smaller experiment; the demonstration separately checks fixed full-size stress behaviour. Source fingerprints include all installed package `.py` files. Configuration and data are separately hashed, and installed dependency versions are exported.

Future extensions could implement rolling-origin validation, repeated-borrower grouped resampling, external schema-validated data ingestion and model registry integrations. These should preserve feature availability, maturity and final holdout isolation.
