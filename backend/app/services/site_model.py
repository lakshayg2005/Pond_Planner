"""Learned pond-site model: a gradient-boosted tree ensemble, run with NumPy.

The model is trained offline (ml/train_site_model.py) with scikit-learn and
exported to backend/app/models/site_model.json. Evaluating the exported trees
here keeps scikit-learn (and its memory) off the 512 MB server.

Set POND_RANKING=heuristic to rank with the hand-weighted formula instead.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .site_features import FEATURE_NAMES

MODEL_PATH = Path(
    os.getenv(
        "POND_SITE_MODEL",
        Path(__file__).resolve().parents[1] / "models" / "site_model.json",
    )
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Tree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    missing_left: np.ndarray
    is_leaf: np.ndarray
    value: np.ndarray
    depth: int


@dataclass(frozen=True, slots=True)
class SiteModel:
    baseline: float
    trees: tuple[_Tree, ...]
    version: str
    # (west, south, east, north) of the area the model was validated for.
    bounds_lonlat: tuple[float, float, float, float] | None = None

    def covers(self, longitude: float, latitude: float) -> bool:
        if self.bounds_lonlat is None:
            return True
        west, south, east, north = self.bounds_lonlat
        return west <= longitude <= east and south <= latitude <= north

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probability that each row of `features` is a good pond site."""
        x = np.asarray(features, dtype=np.float64)
        rows = np.arange(len(x))
        raw = np.full(len(x), self.baseline)
        for tree in self.trees:
            node = np.zeros(len(x), dtype=np.int64)
            for _ in range(tree.depth + 1):
                value = x[rows, tree.feature[node]]
                go_left = np.where(
                    np.isnan(value),
                    tree.missing_left[node],
                    value <= tree.threshold[node],
                )
                child = np.where(go_left, tree.left[node], tree.right[node])
                node = np.where(tree.is_leaf[node], node, child)
            raw += tree.value[node]
        return 1.0 / (1.0 + np.exp(-raw))


def load_model(path: Path = MODEL_PATH) -> SiteModel:
    data = json.loads(Path(path).read_text())
    if tuple(data["features"]) != FEATURE_NAMES:
        raise ValueError("Site model was trained on different features")
    trees = tuple(
        _Tree(
            feature=np.asarray(t["feature"], dtype=np.int64),
            threshold=np.asarray(t["threshold"], dtype=np.float64),
            left=np.asarray(t["left"], dtype=np.int64),
            right=np.asarray(t["right"], dtype=np.int64),
            missing_left=np.asarray(t["missing_left"], dtype=bool),
            is_leaf=np.asarray(t["is_leaf"], dtype=bool),
            value=np.asarray(t["value"], dtype=np.float64),
            depth=int(t["depth"]),
        )
        for t in data["trees"]
    )
    bounds = data.get("bounds_lonlat")
    return SiteModel(
        baseline=float(data["baseline"]),
        trees=trees,
        version=data["version"],
        bounds_lonlat=tuple(bounds) if bounds else None,
    )


@lru_cache(maxsize=1)
def get_model() -> SiteModel | None:
    """The deployed model, or None to fall back to the hand-weighted formula."""
    if os.getenv("POND_RANKING", "learned").lower() == "heuristic":
        return None
    try:
        model = load_model()
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Site model unavailable (%s); using the heuristic ranking", exc)
        return None
    logger.info("Loaded site model %s (%d trees)", model.version, len(model.trees))
    return model
