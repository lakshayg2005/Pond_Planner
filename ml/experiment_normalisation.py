"""Compare raw features with features ranked within each analysed area.

    python ml/experiment_normalisation.py

Same samples, same leave-one-region-out split; only the feature encoding
changes. Kept in the repo so the comparison can be re-run.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

from app.services.site_features import FEATURE_NAMES, rank_within_area  # noqa: E402
from train_site_model import make_model  # noqa: E402

DATA = HERE / "data"


def main():
    rows = list(csv.DictReader(open(DATA / "samples_india.csv")))
    raw = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
    ranked = np.zeros_like(raw)
    for tile in sorted({r["tile"] for r in rows}):
        grid = np.load(DATA / "tiles_india" / f"{tile}.npz")
        percentiles = rank_within_area(grid["features"], grid["inside"])
        for i, r in enumerate(rows):
            if r["tile"] == tile:
                ranked[i] = percentiles[int(r["row"]), int(r["col"])]

    y = np.array([int(r["label"]) for r in rows])
    regions = np.array([r["region"] for r in rows])
    heuristic = np.array([float(r["heuristic_score"]) for r in rows])

    for name, x in (("raw", raw), ("ranked within area", ranked)):
        oof = np.full(len(y), np.nan)
        per_region = []
        for region in sorted(set(regions)):
            test = regions == region
            if y[test].sum() < 10:
                continue
            oof[test] = make_model().fit(x[~test], y[~test]).predict_proba(x[test])[:, 1]
            per_region.append(f"{region}={roc_auc_score(y[test], oof[test]):.2f}")
        scored = ~np.isnan(oof)
        print(f"{name:20s} pooled AUC {roc_auc_score(y[scored], oof[scored]):.3f} | "
              + " ".join(per_region))
    scored = ~np.isnan(oof)
    print(f"{'heuristic formula':20s} pooled AUC {roc_auc_score(y[scored], heuristic[scored]):.3f}")


if __name__ == "__main__":
    main()
