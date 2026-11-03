"""CI acceptance checks: scenario outputs must actually identify introduced failures."""

import json
import sys
from pathlib import Path


def check(directory):
    root = Path(directory)
    summary = json.loads((root / "suite_summary.json").read_text())
    defects = {"leakage", "immature", "duplicate", "unavailable_feature"}
    statuses = {entry["scenario"]: entry["status"] for entry in summary}
    for scenario in defects:
        if statuses.get(scenario) != "detected_defect":
            raise AssertionError(f"Integrity defect was not caught: {scenario}")
    for scenario in ["mechanism_shift", "calibration_error", "downturn"]:
        validation = json.loads((root / scenario / "validation.json").read_text())
        if not any(f["code"] == "CALIBRATION_BIAS" for f in validation["findings"]):
            raise AssertionError(f"Expected calibration failure not detected: {scenario}")
    for scenario in ["composition_shift", "downturn"]:
        validation = json.loads((root / scenario / "validation.json").read_text())
        if not any(f["code"] == "POPULATION_SHIFT" for f in validation["findings"]):
            raise AssertionError(f"Expected population shift not detected: {scenario}")
    print(
        "Demonstration acceptance passed: integrity, calibration and population failures detected."
    )


if __name__ == "__main__":
    check(sys.argv[1])
