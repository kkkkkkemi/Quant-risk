# Development and Git workflow

1. Install the package with development dependencies in a virtual environment.
2. Create a branch, for example `git switch -c feature/rolling-validation`.
3. Make one coherent modelling or engineering change. Preserve the final-test boundary.
4. Run relevant tests, `ruff check .`, `ruff format --check .`, and a demonstration when numerical behaviour changes.
5. Commit source and documentation. Keep generated data, PDF bundles and virtual environments out of Git.
6. Push the branch and open a pull request using the supplied template. Record checks and numerical implications.
7. Review the Actions artifacts, especially affected populations and new calibration warnings. Merge only after review.

To publish this local project, create an empty GitHub repository, configure its remote, and push the branch. Do not describe the Actions deliverable as successful until an actual hosted run completes. Local checks can verify the same commands but cannot substitute for GitHub evidence.

Use fresh output directories for experiments and include source/configuration fingerprints when comparing results. Do not tune repeatedly against the fixed final test after observing its performance; use a newly generated future evaluation window for new model selection decisions.
