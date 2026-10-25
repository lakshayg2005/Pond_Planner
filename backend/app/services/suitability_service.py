# from __future__ import annotations

# from dataclasses import dataclass

# import numpy as np


# @dataclass(slots=True)
# class PondCandidate:
#     """A potential pond outlet/location."""

#     row: int
#     col: int
#     elevation_m: float
#     slope_percent: float
#     flow_accumulation: int
#     score: float


# def _normalize(values: np.ndarray) -> np.ndarray:
#     """
#     Min-max normalize values to [0, 1].

#     Constant arrays are mapped to zero.
#     """
#     values = np.asarray(values, dtype=float)

#     minimum = np.min(values)
#     maximum = np.max(values)

#     if maximum <= minimum:
#         return np.zeros_like(values, dtype=float)

#     return (values - minimum) / (maximum - minimum)


# def calculate_candidate_score(
#     slope_percent: np.ndarray,
#     flow_accumulation: np.ndarray,
#     max_slope_percent: float = 8.0,
# ) -> np.ndarray:
#     """
#     Calculate a preliminary pond suitability score.

#     Higher flow accumulation is preferred.

#     Lower slope is preferred, but slopes above max_slope_percent
#     are considered unsuitable.

#     Score:

#         60% flow accumulation
#         40% slope suitability

#     This is a screening score, not an engineering design score.
#     """

#     slope = np.asarray(slope_percent, dtype=float)
#     accumulation = np.asarray(flow_accumulation, dtype=float)

#     if slope.shape != accumulation.shape:
#         raise ValueError(
#             "Slope and flow accumulation must have the same shape"
#         )

#     if max_slope_percent <= 0:
#         raise ValueError(
#             "max_slope_percent must be greater than zero"
#         )

#     # Log transformation prevents extremely large accumulation
#     # values from dominating the entire score.
#     accumulation_score = _normalize(
#         np.log1p(np.maximum(accumulation, 0))
#     )

#     # Flat terrain receives the highest slope score.
#     # max_slope_percent receives zero.
#     slope_score = 1.0 - np.clip(
#         slope / max_slope_percent,
#         0.0,
#         1.0,
#     )

#     score = (
#         0.60 * accumulation_score
#         + 0.40 * slope_score
#     )

#     # Anything above the configured slope threshold is unsuitable.
#     score = np.where(
#         slope <= max_slope_percent,
#         score,
#         0.0,
#     )

#     return score


# def detect_pond_candidates(
#     elevation_m: np.ndarray,
#     slope_percent: np.ndarray,
#     flow_accumulation: np.ndarray,
#     *,
#     max_slope_percent: float = 8.0,
#     minimum_accumulation: int = 10,
#     max_candidates: int = 10,
#     minimum_distance_cells: int = 10,
# ) -> list[PondCandidate]:
#     """
#     Detect and rank preliminary pond candidates.

#     Candidates must satisfy:

#         slope <= max_slope_percent
#         flow accumulation >= minimum_accumulation

#     Non-maximum suppression is then applied so that several
#     candidates do not appear immediately next to each other.
#     """

#     elevation = np.asarray(elevation_m, dtype=float)
#     slope = np.asarray(slope_percent, dtype=float)
#     accumulation = np.asarray(flow_accumulation)

#     if not (
#         elevation.shape
#         == slope.shape
#         == accumulation.shape
#     ):
#         raise ValueError(
#             "Elevation, slope, and flow accumulation "
#             "must have identical shapes"
#         )

#     if max_candidates <= 0:
#         raise ValueError("max_candidates must be greater than zero")

#     if minimum_accumulation < 1:
#         raise ValueError(
#             "minimum_accumulation must be at least 1"
#         )

#     if minimum_distance_cells < 0:
#         raise ValueError(
#             "minimum_distance_cells cannot be negative"
#         )

#     score = calculate_candidate_score(
#         slope,
#         accumulation,
#         max_slope_percent=max_slope_percent,
#     )

#     eligible = (
#         (slope <= max_slope_percent)
#         & (accumulation >= minimum_accumulation)
#         & np.isfinite(elevation)
#         & np.isfinite(slope)
#         & np.isfinite(accumulation)
#         & (score > 0)
#     )

#     rows, cols = elevation.shape

#     candidate_indices = np.argwhere(eligible)

#     # Sort by score from highest to lowest.
#     candidate_indices = sorted(
#         candidate_indices.tolist(),
#         key=lambda rc: score[rc[0], rc[1]],
#         reverse=True,
#     )

#     selected: list[PondCandidate] = []

#     for row, col in candidate_indices:

#         # Non-maximum suppression:
#         # don't select candidates too close to an already
#         # selected candidate.
#         too_close = False

#         for existing in selected:
#             distance = np.hypot(
#                 row - existing.row,
#                 col - existing.col,
#             )

#             if distance < minimum_distance_cells:
#                 too_close = True
#                 break

#         if too_close:
#             continue

#         selected.append(
#             PondCandidate(
#                 row=int(row),
#                 col=int(col),
#                 elevation_m=float(elevation[row, col]),
#                 slope_percent=float(slope[row, col]),
#                 flow_accumulation=int(accumulation[row, col]),
#                 score=float(score[row, col]),
#             )
#         )

#         if len(selected) >= max_candidates:
#             break

#     return selected
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# D8 direction encoding used by flow_service.py
# 1  = NE
# 2  = E
# 4  = SE
# 8  = S
# 16 = SW
# 32 = W
# 64 = NW
# 128 = N
#
# 0 = no flow


@dataclass(frozen=True)
class PondCandidate:
    row: int
    col: int
    elevation_m: float
    slope_percent: float
    flow_accumulation: int
    score: float

    # Advanced diagnostics
    basin_score: float = 0.0
    storage_score: float = 0.0
    channel_penalty: float = 0.0
    non_channel_score: float = 1.0
    local_relief_score: float = 0.0
    reason: str = ""


def _normalize(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an array to [0, 1].
    """
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(values)

    if not np.any(finite):
        return np.zeros_like(values, dtype=float)

    result = np.zeros_like(values, dtype=float)

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if maximum <= minimum:
        result[finite] = 1.0
        return result

    result[finite] = (
        values[finite] - minimum
    ) / (maximum - minimum)

    return np.clip(result, 0.0, 1.0)


def _local_mean(
    array: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """
    Calculate local mean using a square moving window.

    Uses only NumPy so no new dependency is required.
    """
    array = np.asarray(array, dtype=float)

    padded = np.pad(
        array,
        radius,
        mode="edge",
    )

    result = np.zeros_like(array, dtype=float)

    size = 2 * radius + 1

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            window = padded[
                row : row + size,
                col : col + size,
            ]

            finite = window[np.isfinite(window)]

            if finite.size:
                result[row, col] = np.mean(finite)
            else:
                result[row, col] = np.nan

    return result


def _local_max(
    array: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """
    Calculate local maximum using a square window.
    """
    array = np.asarray(array, dtype=float)

    padded = np.pad(
        array,
        radius,
        mode="edge",
    )

    result = np.zeros_like(array, dtype=float)

    size = 2 * radius + 1

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            window = padded[
                row : row + size,
                col : col + size,
            ]

            finite = window[np.isfinite(window)]

            if finite.size:
                result[row, col] = np.max(finite)
            else:
                result[row, col] = np.nan

    return result


def calculate_candidate_score(
    slope_percent: np.ndarray,
    flow_accumulation: np.ndarray,
    max_slope_percent: float = 8.0,
) -> np.ndarray:
    """
    Backward-compatible basic suitability score.

    High accumulation + low slope = high score.

    This function is intentionally kept compatible with
    the existing unit tests.
    """
    slope = np.asarray(
        slope_percent,
        dtype=float,
    )

    accumulation = np.asarray(
        flow_accumulation,
        dtype=float,
    )

    if slope.shape != accumulation.shape:
        raise ValueError(
            "slope_percent and flow_accumulation "
            "must have the same shape"
        )

    slope_score = np.clip(
        1.0 - slope / max_slope_percent,
        0.0,
        1.0,
    )

    # Log scaling prevents very large accumulation values
    # from completely dominating the score.
    accumulation_score = _normalize(
        np.log1p(np.maximum(accumulation, 0.0))
    )

    score = (
        0.35 * slope_score
        + 0.65 * accumulation_score
    )

    score[slope > max_slope_percent] = 0.0

    return np.clip(score, 0.0, 1.0)


def _calculate_channel_likelihood(
    accumulation: np.ndarray,
    flow_direction: Optional[np.ndarray],
) -> np.ndarray:
    """
    Estimate channel/river likelihood.

    Important:
    High flow accumulation alone does NOT mean that a cell is a river.
    A pond outlet can naturally have high accumulation.

    When flow direction is unavailable, accumulation is used only as a
    soft signal and never as sufficient evidence for hard rejection.

    When D8 flow direction is available, concentrated flowing cells
    receive stronger channel likelihood.
    """
    acc = np.asarray(accumulation, dtype=float)

    log_acc = np.log1p(np.maximum(acc, 0.0))
    global_acc = _normalize(log_acc)

    local_acc = _local_mean(
        log_acc,
        radius=1,
    )
    local_acc = _normalize(local_acc)

    if flow_direction is None:
        # Without actual drainage connectivity, high accumulation
        # should NOT be interpreted as a river.
        #
        # Keep only a weak soft penalty.
        return np.clip(
            0.20 * global_acc + 0.10 * local_acc,
            0.0,
            0.30,
        )

    flow = np.asarray(flow_direction)

    if flow.shape != acc.shape:
        raise ValueError(
            "flow_direction and flow_accumulation "
            "must have the same shape"
        )

    flowing = flow != 0

    # Strong accumulation + actual flowing cell.
    flow_channel = (
        0.70 * global_acc
        + 0.30 * local_acc
    )

    likelihood = np.where(
        flowing,
        flow_channel,
        0.15 * global_acc,
    )

    return np.clip(
        likelihood,
        0.0,
        1.0,
    )


def _calculate_basin_storage_score(
    elevation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate local basin/storage characteristics.

    A good pond location should generally be lower than its
    surrounding terrain.

    Returns:
        basin_score
        local_relief_score
    """
    elevation = np.asarray(
        elevation,
        dtype=float,
    )

    local_mean = _local_mean(
        elevation,
        radius=1,
    )

    local_max = _local_max(
        elevation,
        radius=1,
    )

    # How much lower is the candidate than nearby terrain?
    depression_depth = np.maximum(
        local_mean - elevation,
        0.0,
    )

    local_relief = np.maximum(
        local_max - elevation,
        0.0,
    )

    basin_score = _normalize(
        depression_depth
    )

    local_relief_score = _normalize(
        local_relief
    )

    return (
        np.clip(basin_score, 0.0, 1.0),
        np.clip(local_relief_score, 0.0, 1.0),
    )


def _build_reason(
    water_supply: float,
    basin_score: float,
    storage_score: float,
    channel_penalty: float,
    slope_score: float,
) -> str:
    """
    Human-readable explanation for a candidate.
    """
    reasons = []

    if water_supply >= 0.70:
        reasons.append("high runoff supply")
    elif water_supply >= 0.40:
        reasons.append("moderate runoff supply")

    if basin_score >= 0.70:
        reasons.append("strong local depression")

    if storage_score >= 0.70:
        reasons.append("good terrain storage potential")

    if slope_score >= 0.70:
        reasons.append("low slope")

    if channel_penalty <= 0.25:
        reasons.append("low channel likelihood")
    elif channel_penalty >= 0.70:
        reasons.append("strong drainage/channel signature")

    if not reasons:
        return "Mixed terrain suitability"

    return ", ".join(reasons)


@dataclass(slots=True)
class HeuristicScores:
    """Per-cell grids from the hand-weighted suitability formula."""

    score: np.ndarray            # 0 where the cell is not eligible
    valid: np.ndarray            # eligibility (slope, accumulation, channel)
    water_supply: np.ndarray
    slope_score: np.ndarray
    basin_score: np.ndarray
    local_relief_score: np.ndarray
    channel_penalty: np.ndarray
    non_channel_score: np.ndarray


def heuristic_scores(
    elevation_m: np.ndarray,
    slope_percent: np.ndarray,
    flow_accumulation: np.ndarray,
    max_slope_percent: float = 8.0,
    minimum_accumulation: int = 10,
    flow_direction: Optional[np.ndarray] = None,
) -> HeuristicScores:
    """
    Hand-weighted pond-site suitability for every cell.

    The algorithm combines:

        1. runoff supply
        2. low slope
        3. local depression
        4. local terrain relief
        5. channel/river likelihood

    Strong drainage cells are rejected rather than merely
    receiving a lower score.

    Existing callers remain compatible because flow_direction
    is optional.
    """
    elevation = np.asarray(
        elevation_m,
        dtype=float,
    )

    slope = np.asarray(
        slope_percent,
        dtype=float,
    )

    accumulation = np.asarray(
        flow_accumulation,
        dtype=float,
    )

    if not (
        elevation.shape
        == slope.shape
        == accumulation.shape
    ):
        raise ValueError(
            "elevation_m, slope_percent, and "
            "flow_accumulation must have the same shape"
        )

    if flow_direction is not None:
        flow_direction = np.asarray(
            flow_direction
        )

        if flow_direction.shape != elevation.shape:
            raise ValueError(
                "flow_direction must have the same shape "
                "as elevation_m"
            )

    valid = (
        np.isfinite(elevation)
        & np.isfinite(slope)
        & np.isfinite(accumulation)
    )

    valid &= slope <= max_slope_percent
    valid &= accumulation >= minimum_accumulation

    # ---------------------------------------------------------
    # 1. Water supply
    # ---------------------------------------------------------

    water_supply = _normalize(
        np.log1p(
            np.maximum(accumulation, 0.0)
        )
    )

    # ---------------------------------------------------------
    # 2. Low slope
    # ---------------------------------------------------------

    slope_score = np.clip(
        1.0
        - slope / max_slope_percent,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # 3. Basin / storage evidence
    # ---------------------------------------------------------

    basin_score, local_relief_score = (
        _calculate_basin_storage_score(
            elevation
        )
    )

    # ---------------------------------------------------------
    # 4. Channel likelihood
    # ---------------------------------------------------------

    channel_likelihood = (
        _calculate_channel_likelihood(
            accumulation,
            flow_direction,
        )
    )

    channel_penalty = channel_likelihood

    non_channel_score = 1.0 - channel_penalty

    # ---------------------------------------------------------
    # 5. Combined score
    # ---------------------------------------------------------

    # score = (
    #     0.30 * water_supply
    #     + 0.25 * basin_score
    #     + 0.20 * slope_score
    #     + 0.15 * local_relief_score
    #     + 0.10 * non_channel_score
    # )
    score = (
        0.30 * water_supply
        + 0.30 * basin_score
        + 0.15 * slope_score
        + 0.15 * local_relief_score
        + 0.10 * non_channel_score
    )

    # Hard rejection:
    #
    # Very strong drainage/channel cells should not become
    # pond candidates merely because they have huge runoff.
    #
    # Threshold deliberately leaves room for natural drainage
    # receiving areas while removing the strongest channels.
    # strong_channel = (
    #     channel_penalty >= 0.90
    # )

    # valid &= ~strong_channel
    # Only perform hard channel rejection when actual D8 flow
    # information is available.
    #
    # Without flow direction, accumulation alone is insufficient
    # evidence to classify a location as a river.
    if flow_direction is not None:
      strong_channel = channel_penalty >= 0.90
      valid &= ~strong_channel

    score[~valid] = 0.0

    return HeuristicScores(
        score=score,
        valid=valid,
        water_supply=water_supply,
        slope_score=slope_score,
        basin_score=basin_score,
        local_relief_score=local_relief_score,
        channel_penalty=channel_penalty,
        non_channel_score=non_channel_score,
    )


def detect_pond_candidates(
    elevation_m: np.ndarray,
    slope_percent: np.ndarray,
    flow_accumulation: np.ndarray,
    max_slope_percent: float = 8.0,
    minimum_accumulation: int = 10,
    max_candidates: int = 10,
    minimum_distance_cells: int = 10,
    flow_direction: Optional[np.ndarray] = None,
    learned_score: Optional[np.ndarray] = None,
) -> list[PondCandidate]:
    """
    Rank pond sites and keep the best ones, spaced apart.

    Cells are eligible only if they pass the hard rules of the
    hand-weighted formula (slope limit, minimum runoff, not a strong
    channel). They are then ranked by `learned_score` (the trained
    site model's probability) when given, else by the formula's score.
    """
    elevation = np.asarray(elevation_m, dtype=float)
    slope = np.asarray(slope_percent, dtype=float)
    accumulation = np.asarray(flow_accumulation, dtype=float)

    grids = heuristic_scores(
        elevation,
        slope,
        accumulation,
        max_slope_percent=max_slope_percent,
        minimum_accumulation=minimum_accumulation,
        flow_direction=flow_direction,
    )

    if not np.any(grids.valid):
        return []

    if learned_score is not None:
        score = np.where(grids.valid, np.asarray(learned_score, dtype=float), 0.0)
    else:
        score = grids.score

    water_supply = grids.water_supply
    slope_score = grids.slope_score
    basin_score = grids.basin_score
    local_relief_score = grids.local_relief_score
    channel_penalty = grids.channel_penalty
    non_channel_score = grids.non_channel_score

    # ---------------------------------------------------------
    # 6. Candidate extraction
    # ---------------------------------------------------------

    rows, cols = np.where(
        score > 0.0
    )

    candidates = []

    for row, col in zip(rows, cols):
        candidates.append(
            PondCandidate(
                row=int(row),
                col=int(col),
                elevation_m=float(
                    elevation[row, col]
                ),
                slope_percent=float(
                    slope[row, col]
                ),
                flow_accumulation=int(
                    accumulation[row, col]
                ),
                score=float(
                    score[row, col]
                ),
                basin_score=float(
                    basin_score[row, col]
                ),
                storage_score=float(
                    local_relief_score[row, col]
                ),
                channel_penalty=float(
                    channel_penalty[row, col]
                ),
                non_channel_score=float(
                    non_channel_score[row, col]
                ),
                local_relief_score=float(
                    local_relief_score[row, col]
                ),
                reason=(
                    f"learned suitability {score[row, col]:.2f}; "
                    if learned_score is not None
                    else ""
                ) + _build_reason(
                    float(water_supply[row, col]),
                    float(basin_score[row, col]),
                    float(local_relief_score[row, col]),
                    float(channel_penalty[row, col]),
                    float(slope_score[row, col]),
                ),
            )
        )

    # Highest suitability first.
    candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    # ---------------------------------------------------------
    # 7. Spatial non-maximum suppression
    # ---------------------------------------------------------

    selected: list[PondCandidate] = []

    for candidate in candidates:
        too_close = False

        for existing in selected:
            distance = np.sqrt(
                (
                    candidate.row
                    - existing.row
                ) ** 2
                + (
                    candidate.col
                    - existing.col
                ) ** 2
            )

            if (
                distance
                < minimum_distance_cells
            ):
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

        if len(selected) >= max_candidates:
            break

    return selected