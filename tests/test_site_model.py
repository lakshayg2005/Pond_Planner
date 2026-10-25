import json

import numpy as np
import pytest

from app.services import site_model
from app.services.site_features import FEATURE_NAMES, compute_feature_grid
from app.services.suitability_service import detect_pond_candidates


def _valley(rows=60, cols=60, resolution=20.0):
    y, x = np.mgrid[0:rows, 0:cols]
    elevation = 100.0 + 0.5 * np.abs(x - cols / 2) + 0.05 * y
    slope = np.gradient(elevation, resolution)
    slope = np.hypot(*slope) * 100.0
    accumulation = np.ones((rows, cols))
    accumulation[:, cols // 2] = np.arange(1, rows + 1) * 10
    return elevation, slope, accumulation, resolution


def test_feature_grid_shape_and_values():
    elevation, slope, accumulation, resolution = _valley()
    features = compute_feature_grid(elevation, slope, accumulation, resolution)

    assert features.shape == (*elevation.shape, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    tpi_500 = features[..., FEATURE_NAMES.index("tpi_500m")]
    # The valley floor sits below its surroundings, the slopes above.
    assert tpi_500[30, 30] < 0 < tpi_500[30, 5]


def test_exported_trees_match_scikit_learn(tmp_path):
    sklearn = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(FEATURE_NAMES)))
    y = (x[:, 0] + 0.5 * x[:, 3] > 0).astype(int)
    model = sklearn.HistGradientBoostingClassifier(max_iter=20, max_depth=3).fit(x, y)

    trees = [
        {
            "feature": t[0].nodes["feature_idx"].tolist(),
            "threshold": t[0].nodes["num_threshold"].tolist(),
            "left": t[0].nodes["left"].tolist(),
            "right": t[0].nodes["right"].tolist(),
            "missing_left": t[0].nodes["missing_go_to_left"].astype(bool).tolist(),
            "is_leaf": t[0].nodes["is_leaf"].astype(bool).tolist(),
            "value": t[0].nodes["value"].tolist(),
            "depth": int(t[0].nodes["depth"].max()),
        }
        for t in model._predictors
    ]
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "features": list(FEATURE_NAMES),
                "baseline": float(np.ravel(model._baseline_prediction)[0]),
                "trees": trees,
            }
        )
    )

    exported = site_model.load_model(path)
    np.testing.assert_allclose(
        exported.predict_proba(x), model.predict_proba(x)[:, 1], atol=1e-9
    )


def test_learned_score_ranks_but_rules_still_apply():
    elevation, slope, accumulation, _ = _valley()
    learned = np.zeros(elevation.shape)
    learned[10, 30] = 0.9      # eligible valley cell
    learned[30, 5] = 0.99      # high score but on the steep side: not eligible

    candidates = detect_pond_candidates(
        elevation,
        slope,
        accumulation,
        max_slope_percent=8.0,
        minimum_accumulation=10,
        max_candidates=1,
        learned_score=learned,
    )

    assert (candidates[0].row, candidates[0].col) == (10, 30)
    assert candidates[0].score == pytest.approx(0.9)


def test_heuristic_ranking_can_be_forced(monkeypatch):
    monkeypatch.setenv("POND_RANKING", "heuristic")
    site_model.get_model.cache_clear()
    try:
        assert site_model.get_model() is None
    finally:
        site_model.get_model.cache_clear()
