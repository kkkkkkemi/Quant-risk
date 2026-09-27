"""Compare two independent runs; metadata and PDF timestamps are intentionally excluded."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(first, second):
    first, second = Path(first), Path(second)
    a, b = pd.read_csv(first / "predictions.csv"), pd.read_csv(second / "predictions.csv")
    pd.testing.assert_frame_equal(a.drop(columns="predicted_pd"), b.drop(columns="predicted_pd"))
    np.testing.assert_allclose(a.predicted_pd, b.predicted_pd, atol=1e-10, rtol=0)
    va = json.loads((first / "validation.json").read_text())
    vb = json.loads((second / "validation.json").read_text())
    numeric = ["roc_auc", "gini", "brier", "log_loss", "mean_pd", "observed_rate", "pd_gap"]
    for left, right in zip(va["metrics"], vb["metrics"], strict=True):
        for key in numeric:
            if left[key] is None:
                assert right[key] is None
            else:
                np.testing.assert_allclose(left[key], right[key], atol=1e-10, rtol=0)
    print("Predictions and point metrics agree within absolute tolerance 1e-10.")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
