"""Train, evaluate and export the pond-site model.

    python ml/train_site_model.py india          # all-India study (no export)
    python ml/train_site_model.py chhattisgarh   # the deployed model

india: leave-one-region-out over 8 regions. Kept as a record of why the
deployed model is limited to one state (accuracy does not transfer).

chhattisgarh: leave-one-district-out over the training districts, then the
final model (trained on all training districts) is scored once on held-out
districts that were fixed in ml/regions.py before any results were seen.
Only this model is exported, to backend/app/models/site_model.json.

In both cases the hand-weighted formula the app used before is scored on
exactly the same points.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

from app.services.site_features import FEATURE_NAMES  # noqa: E402
from app.services.site_model import load_model  # noqa: E402
from regions import CHHATTISGARH_BOUNDS, CHHATTISGARH_TEST  # noqa: E402

MODEL_OUT = HERE.parent / "backend" / "app" / "models" / "site_model.json"
TOP_K = 10
MIN_DISTANCE_CELLS = 10       # same spacing the app uses between sites
HIT_DISTANCE_M = 100.0        # a pick "lands on a pond" within this distance
MIN_TEST_POSITIVES = 10       # regions with fewer ponds are only used for training


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=0,
    )


def load_samples(region_set: str):
    with open(DATA / f"samples_{region_set}.csv") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    heuristic = np.array([float(r["heuristic_score"]) for r in rows])
    regions = np.array([r["region"] for r in rows])
    return x, y, heuristic, regions


def top_k_sites(score: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    """Best cells, at least MIN_DISTANCE_CELLS apart (as the app selects them)."""
    flat = np.where(eligible, score, -np.inf).ravel()
    order = np.argsort(flat)[::-1][:5000]
    chosen: list[tuple[int, int]] = []
    for index in order:
        if not np.isfinite(flat[index]) or flat[index] <= 0:
            break
        row, col = divmod(int(index), score.shape[1])
        if all((row - r) ** 2 + (col - c) ** 2 >= MIN_DISTANCE_CELLS**2 for r, c in chosen):
            chosen.append((row, col))
        if len(chosen) == TOP_K:
            break
    return chosen


def tile_hits(model, tiles_dir: Path, region: str) -> np.ndarray:
    """[model hits, model picks, formula hits, formula picks, chance, tiles]."""
    totals = np.zeros(6)
    for path in sorted(tiles_dir.glob(f"{region}_*.npz")):
        tile = np.load(path)
        eligible = tile["valid"] & tile["inside"]
        if not eligible.any():
            continue
        near_pond = tile["pond_distance"] <= HIT_DISTANCE_M
        prob = np.zeros(eligible.shape)
        prob[eligible] = model.predict_proba(tile["features"][eligible])[:, 1]
        for offset, score in ((0, prob), (2, tile["heuristic"])):
            picks = top_k_sites(score, eligible)
            totals[offset] += sum(bool(near_pond[r, c]) for r, c in picks)
            totals[offset + 1] += len(picks)
        totals[4] += near_pond[eligible].mean()
        totals[5] += 1
    return totals


def summarise(y, model_score, heuristic, hits) -> dict:
    return {
        "samples": int(len(y)),
        "ponds": int(y.sum()),
        "model_roc_auc": round(roc_auc_score(y, model_score), 3),
        "heuristic_roc_auc": round(roc_auc_score(y, heuristic), 3),
        "model_pr_auc": round(average_precision_score(y, model_score), 3),
        "heuristic_pr_auc": round(average_precision_score(y, heuristic), 3),
        "pond_share": round(float(y.mean()), 3),
        "top10_model_hit_rate": round(hits[0] / max(hits[1], 1), 3),
        "top10_heuristic_hit_rate": round(hits[2] / max(hits[3], 1), 3),
        "top10_chance_rate": round(hits[4] / max(hits[5], 1), 3),
        "top10_picks": int(hits[1]),
        "tiles": int(hits[5]),
    }


def cross_validate(x, y, heuristic, regions, tiles_dir) -> tuple[dict, dict]:
    oof = np.full(len(y), np.nan)
    hits = np.zeros(6)
    per_region = {}
    for region in sorted(set(regions)):
        test = regions == region
        if y[test].sum() < MIN_TEST_POSITIVES or (y[test] == 0).sum() == 0:
            continue
        model = make_model().fit(x[~test], y[~test])
        oof[test] = model.predict_proba(x[test])[:, 1]
        region_hits = tile_hits(model, tiles_dir, region)
        hits += region_hits
        per_region[region] = summarise(y[test], oof[test], heuristic[test], region_hits)
        print(region, per_region[region])
    scored = ~np.isnan(oof)
    overall = summarise(y[scored], oof[scored], heuristic[scored], hits)
    overall["evaluation"] = "leave-one-region-out"
    return overall, per_region


def export(model, metrics: dict) -> None:
    trees = []
    for iteration in model._predictors:
        nodes = iteration[0].nodes
        trees.append(
            {
                "feature": nodes["feature_idx"].tolist(),
                "threshold": [float(v) for v in nodes["num_threshold"]],
                "left": nodes["left"].tolist(),
                "right": nodes["right"].tolist(),
                "missing_left": nodes["missing_go_to_left"].astype(bool).tolist(),
                "is_leaf": nodes["is_leaf"].astype(bool).tolist(),
                "value": [float(v) for v in nodes["value"]],
                "depth": int(nodes["depth"].max()),
            }
        )
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(
        json.dumps(
            {
                "version": f"hgb-chhattisgarh-{date.today().isoformat()}",
                "features": list(FEATURE_NAMES),
                "bounds_lonlat": list(CHHATTISGARH_BOUNDS),
                "baseline": float(np.ravel(model._baseline_prediction)[0]),
                "metrics": metrics,
                "trees": trees,
            }
        )
    )


def run_india():
    x, y, heuristic, regions = load_samples("india")
    print(f"india: {len(y)} samples, {y.sum()} ponds, {len(set(regions))} regions")
    overall, per_region = cross_validate(x, y, heuristic, regions, DATA / "tiles_india")
    (HERE / "results_india.json").write_text(
        json.dumps({"overall": overall, "per_region": per_region}, indent=2)
    )
    print(json.dumps(overall, indent=2))


def run_chhattisgarh():
    x, y, heuristic, regions = load_samples("chhattisgarh")
    tiles_dir = DATA / "tiles_chhattisgarh"
    held_out = np.isin(regions, list(CHHATTISGARH_TEST))
    print(
        f"chhattisgarh: train {(~held_out).sum()} samples / {y[~held_out].sum()} ponds, "
        f"test {held_out.sum()} samples / {y[held_out].sum()} ponds"
    )

    # 1. Development: leave-one-district-out within the training districts.
    cv, cv_regions = cross_validate(
        x[~held_out], y[~held_out], heuristic[~held_out], regions[~held_out], tiles_dir
    )

    # 2. Final model on all training districts, scored once on held-out districts.
    final = make_model().fit(x[~held_out], y[~held_out])
    test_hits = np.zeros(6)
    test_regions = {}
    for region in CHHATTISGARH_TEST:
        mask = regions == region
        if not mask.any():
            continue
        hits = tile_hits(final, tiles_dir, region)
        test_hits += hits
        test_regions[region] = summarise(
            y[mask], final.predict_proba(x[mask])[:, 1], heuristic[mask], hits
        )
    test = summarise(
        y[held_out], final.predict_proba(x[held_out])[:, 1], heuristic[held_out], test_hits
    )
    test["evaluation"] = "held-out districts, scored once"

    importance = permutation_importance(
        final, x[~held_out], y[~held_out], scoring="roc_auc", n_repeats=5, random_state=0
    )
    metrics = {
        "cross_validation": cv,
        "held_out_test": test,
        "feature_importance": {
            name: round(float(v), 4)
            for name, v in sorted(
                zip(FEATURE_NAMES, importance.importances_mean), key=lambda t: -t[1]
            )
        },
    }
    export(final, metrics)
    exported = load_model(MODEL_OUT)
    difference = np.abs(exported.predict_proba(x) - final.predict_proba(x)[:, 1]).max()
    assert difference < 1e-9, f"exported model disagrees with scikit-learn ({difference})"

    (HERE / "results_chhattisgarh.json").write_text(
        json.dumps(
            {**metrics, "cv_per_region": cv_regions, "test_per_region": test_regions},
            indent=2,
        )
    )
    print("cross-validation:", json.dumps(cv, indent=2))
    print("HELD-OUT TEST:", json.dumps(test, indent=2))
    print(json.dumps(test_regions, indent=2))
    print(f"model: {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    {"india": run_india, "chhattisgarh": run_chhattisgarh}[
        sys.argv[1] if len(sys.argv) > 1 else "chhattisgarh"
    ]()
