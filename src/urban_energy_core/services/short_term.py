from __future__ import annotations

import math
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from urban_energy_core.domain.city import City


def _empty_cluster_hourly_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["cluster_label", "hour", "p05", "p50", "p95", "mean", "n_days"]
    )


def _cluster_hourly_stats(X: np.ndarray, raw_labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in sorted(np.unique(raw_labels)):
        members = X[raw_labels == c]
        label = f"cluster_{int(c)}"
        for h in range(members.shape[1]):
            vals = members[:, h]
            rows.append(
                {
                    "cluster_label": label,
                    "hour": int(h),
                    "p05": float(np.percentile(vals, 5)),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                    "mean": float(np.mean(vals)),
                    "n_days": int(members.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _pairwise_dtw_matrix(X: np.ndarray, show_progress: bool = False) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    row_iter = tqdm(range(n), desc="DTW matrix rows", disable=not show_progress, leave=False)
    for i in row_iter:
        for j in range(i + 1, n):
            d = _dtw_distance(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _assign_to_medoids(D: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    return np.argmin(D[:, medoids], axis=1)


def _update_medoids(D: np.ndarray, labels: np.ndarray, k: int, current_medoids: np.ndarray) -> np.ndarray:
    new_medoids = current_medoids.copy()
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        intra = D[np.ix_(members, members)].sum(axis=1)
        new_medoids[c] = members[int(np.argmin(intra))]
    return new_medoids


def _kmedoids(D: np.ndarray, k: int, max_iter: int = 50, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    if k < 1 or k > n:
        raise ValueError("k must be between 1 and n_samples.")
    rng = np.random.default_rng(random_state)
    medoids = np.array(rng.choice(n, size=k, replace=False), dtype=int)

    for _ in range(max_iter):
        labels = _assign_to_medoids(D, medoids)
        new_medoids = _update_medoids(D, labels, k, medoids)
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    labels = _assign_to_medoids(D, medoids)
    return labels, medoids


def _silhouette_from_distance_matrix(D: np.ndarray, labels: np.ndarray) -> float:
    n = D.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0

    sil_vals = np.zeros(n, dtype=float)
    for i in range(n):
        c = labels[i]
        same = np.where(labels == c)[0]
        other_clusters = [u for u in uniq if u != c]

        if len(same) <= 1:
            a_i = 0.0
        else:
            a_i = float(D[i, same[same != i]].mean())

        b_i = np.inf
        for oc in other_clusters:
            members = np.where(labels == oc)[0]
            if len(members):
                b_i = min(b_i, float(D[i, members].mean()))
        if not np.isfinite(b_i):
            sil_vals[i] = 0.0
        else:
            denom = max(a_i, b_i)
            sil_vals[i] = 0.0 if denom == 0 else (b_i - a_i) / denom
    return float(np.mean(sil_vals))


def _daily_profile_matrix(
    load_series: pd.Series,
    min_hours_per_day: int = 20,
    normalize_each_day: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    s = load_series.astype(float).dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("load_series index must be a DatetimeIndex.")

    rows = []
    days = []
    grouped = list(s.groupby(s.index.floor("D")))
    day_iter = tqdm(grouped, desc="Building daily profiles", disable=not show_progress, leave=False)
    for day, day_s in day_iter:
        by_hour = day_s.groupby(day_s.index.hour).mean()
        v = by_hour.reindex(range(24)).to_numpy(dtype=float)
        valid = np.isfinite(v)
        if int(valid.sum()) < min_hours_per_day:
            continue
        if np.any(~valid):
            v = pd.Series(v).interpolate(limit_direction="both").to_numpy(dtype=float)
        if normalize_each_day:
            mu = float(np.mean(v))
            sd = float(np.std(v))
            v = v - mu if sd == 0 else (v - mu) / sd
        rows.append(v)
        days.append(pd.Timestamp(day))

    if not rows:
        return np.empty((0, 24), dtype=float), pd.DatetimeIndex([])
    return np.vstack(rows), pd.DatetimeIndex(days)


def cluster_daily_profiles_dtw(
    load_series: pd.Series,
    k_min: int = 2,
    k_max: int = 6,
    min_days: int = 20,
    min_hours_per_day: int = 20,
    normalize_each_day: bool = True,
    dominance_threshold: float = 0.6,
    random_state: int = 42,
    show_progress: bool = True,
) -> dict:
    """
    DTW cluster daily profiles, select k via silhouette, and decide dominant cluster.
    """
    X, days = _daily_profile_matrix(
        load_series=load_series,
        min_hours_per_day=min_hours_per_day,
        normalize_each_day=normalize_each_day,
        show_progress=show_progress,
    )
    n_days = X.shape[0]
    if n_days == 0:
        return {
            "daily_labels": pd.Series(dtype="string"),
            "fit_summary": {"n_days": 0, "best_k": 0, "silhouette": np.nan},
            "dominant_cluster_label": "no_decision",
            "dominant_cluster_share": np.nan,
            "cluster_hourly_stats": _empty_cluster_hourly_stats(),
        }

    if n_days < min_days:
        labels = pd.Series(["no_decision"] * n_days, index=days, name="dtw_cluster_label", dtype="string")
        return {
            "daily_labels": labels,
            "fit_summary": {"n_days": int(n_days), "best_k": 0, "silhouette": np.nan},
            "dominant_cluster_label": "no_decision",
            "dominant_cluster_share": np.nan,
            "cluster_hourly_stats": _empty_cluster_hourly_stats(),
        }

    D = _pairwise_dtw_matrix(X, show_progress=show_progress)
    k_low = max(2, int(k_min))
    k_high = min(int(k_max), n_days - 1)
    if k_low > k_high:
        labels = pd.Series(["no_decision"] * n_days, index=days, name="dtw_cluster_label", dtype="string")
        return {
            "daily_labels": labels,
            "fit_summary": {"n_days": int(n_days), "best_k": 1, "silhouette": np.nan},
            "dominant_cluster_label": "no_decision",
            "dominant_cluster_share": np.nan,
            "cluster_hourly_stats": _empty_cluster_hourly_stats(),
        }

    best = None
    k_iter = tqdm(range(k_low, k_high + 1), desc="DTW k-search", disable=not show_progress, leave=False)
    for k in k_iter:
        labels_k, medoids_k = _kmedoids(D, k=k, random_state=random_state)
        sil = _silhouette_from_distance_matrix(D, labels_k)
        cand = {"k": k, "labels": labels_k, "medoids": medoids_k, "silhouette": sil}
        if best is None or cand["silhouette"] > best["silhouette"]:
            best = cand

    assert best is not None
    raw_labels = best["labels"]
    label_names = pd.Series([f"cluster_{int(i)}" for i in raw_labels], index=days, name="dtw_cluster_label", dtype="string")
    share = label_names.value_counts(normalize=True).sort_values(ascending=False)
    top_label = str(share.index[0])
    top_share = float(share.iloc[0])
    dominant_label = top_label if top_share >= dominance_threshold else "no_decision"

    labels_out = label_names.copy()
    if dominant_label == "no_decision":
        labels_out = labels_out.astype("string")

    hourly_stats = _cluster_hourly_stats(X, raw_labels)

    return {
        "daily_labels": labels_out,
        "fit_summary": {
            "n_days": int(n_days),
            "best_k": int(best["k"]),
            "silhouette": float(best["silhouette"]),
            "normalize_each_day": bool(normalize_each_day),
        },
        "dominant_cluster_label": dominant_label,
        "dominant_cluster_share": top_share,
        "cluster_hourly_stats": hourly_stats,
    }


def compute_daily_short_term_metrics(
    load_series: pd.Series,
    am_hours: tuple[int, int] = (6, 11),
    pm_hours: tuple[int, int] = (16, 21),
    dtw_labeler: Callable[[pd.Series], str] | None = None,
    dtw_labels: pd.Series | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute daily short-term consumption metrics from an hourly load series.

    Output columns:
    - peak_load
    - p90_top10_mean
    - am_pm_peak_ratio
    - ramp_up_rate
    - dtw_cluster_label
    """
    s = load_series.astype(float).dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("load_series index must be a DatetimeIndex.")
    if s.empty:
        return pd.DataFrame(
            columns=[
                "peak_load",
                "p90_top10_mean",
                "am_pm_peak_ratio",
                "ramp_up_rate",
                "dtw_cluster_label",
            ]
        )

    rows = []
    grouped = list(s.groupby(s.index.floor("D")))
    day_iter = tqdm(grouped, desc="Computing daily metrics", disable=not show_progress)
    for day, day_s in day_iter:
        if day_s.empty:
            continue

        peak_load = float(day_s.max())
        n_top = max(1, int(math.ceil(0.10 * len(day_s))))
        p90_top10_mean = float(day_s.nlargest(n_top).mean())

        am_s = day_s[(day_s.index.hour >= am_hours[0]) & (day_s.index.hour <= am_hours[1])]
        pm_s = day_s[(day_s.index.hour >= pm_hours[0]) & (day_s.index.hour <= pm_hours[1])]
        am_peak = float(am_s.max()) if len(am_s) else np.nan
        pm_peak = float(pm_s.max()) if len(pm_s) else np.nan
        am_pm_peak_ratio = float(am_peak / pm_peak) if np.isfinite(am_peak) and np.isfinite(pm_peak) and pm_peak != 0 else np.nan

        ramp_up_rate = float(day_s.diff().max()) if len(day_s) > 1 else np.nan

        if dtw_labels is not None and pd.Timestamp(day) in dtw_labels.index:
            dtw_cluster_label = dtw_labels.loc[pd.Timestamp(day)]
        else:
            dtw_cluster_label = dtw_labeler(day_s) if dtw_labeler is not None else np.nan

        rows.append(
            {
                "date": pd.Timestamp(day),
                "peak_load": peak_load,
                "p90_top10_mean": p90_top10_mean,
                "am_pm_peak_ratio": am_pm_peak_ratio,
                "ramp_up_rate": ramp_up_rate,
                "dtw_cluster_label": dtw_cluster_label,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "peak_load",
                "p90_top10_mean",
                "am_pm_peak_ratio",
                "ramp_up_rate",
                "dtw_cluster_label",
            ]
        )

    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


def city_short_term_table(
    city: "City",
    per_capita: bool = True,
    weather_normalized: bool = False,
    population_col: str = "Population and dwelling counts / Population, 2021",
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
    am_hours: tuple[int, int] = (6, 11),
    pm_hours: tuple[int, int] = (16, 21),
    dtw_labeler: Callable[[pd.Series], str] | None = None,
    use_dtw_clustering: bool = False,
    dtw_k_min: int = 2,
    dtw_k_max: int = 6,
    dtw_min_days: int = 20,
    dtw_dominance_threshold: float = 0.6,
    winter_only: bool = False,
    winter_months: tuple[int, ...] = (11, 12, 1, 2, 3, 4),
    weekday_only: bool = False,
    aggregate: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute short-term metrics for every FSA in a city.

    If aggregate=True, returns one row per FSA with mean daily metrics.
    If aggregate=False, returns one row per (fsa, date).

    Parameters
    ----------
    winter_only : bool
        If True, keep only data points whose timestamp month is in `winter_months`
        before computing metrics.
    winter_months : tuple[int, ...]
        Month numbers to include when winter_only=True.
    weekday_only : bool
        If True, keep only Monday-Friday timestamps (dayofweek 0..4)
        before computing metrics.
    """
    if weather_normalized and city.weather is None:
        raise ValueError(f"City '{city.name}' has no weather dataframe for weather_normalized=True.")

    records = []
    winter_set = set(int(m) for m in winter_months)
    invalid_months = [m for m in winter_set if m < 1 or m > 12]
    if invalid_months:
        raise ValueError(f"winter_months must be month numbers in 1..12; got invalid: {sorted(invalid_months)}")

    codes = city.list_fsa_codes()
    iter_codes = tqdm(codes, desc=f"Short-term metrics {city.name}") if show_progress else codes
    for code in iter_codes:
        fsa = city.get_fsa(code)
        if fsa.electricity is None:
            continue
        s = fsa.electricity
        if weather_normalized:
            s = fsa.normalize_for_weather(city.weather, dt_col=dt_col, temp_col=temp_col, copy=True)
        if per_capita:
            s = fsa.per_capita_consumption(population_col=population_col)
        if winter_only:
            s = s[s.index.month.isin(winter_set)]
            if s.empty:
                continue
        if weekday_only:
            s = s[s.index.dayofweek <= 4]
            if s.empty:
                continue

        cluster_result = None
        dtw_labels = None
        if use_dtw_clustering:
            cluster_result = cluster_daily_profiles_dtw(
                load_series=s,
                k_min=dtw_k_min,
                k_max=dtw_k_max,
                min_days=dtw_min_days,
                dominance_threshold=dtw_dominance_threshold,
                show_progress=show_progress,
            )
            dtw_labels = cluster_result["daily_labels"]

        daily = compute_daily_short_term_metrics(
            load_series=s,
            am_hours=am_hours,
            pm_hours=pm_hours,
            dtw_labeler=dtw_labeler,
            dtw_labels=dtw_labels,
            show_progress=show_progress,
        )
        if daily.empty:
            continue

        if aggregate:
            rec = {
                "fsa": code,
                "peak_load": float(daily["peak_load"].mean()),
                "p90_top10_mean": float(daily["p90_top10_mean"].mean()),
                "am_pm_peak_ratio": float(daily["am_pm_peak_ratio"].mean()),
                "ramp_up_rate": float(daily["ramp_up_rate"].mean()),
            }
            if use_dtw_clustering and cluster_result is not None:
                rec["dtw_cluster_label"] = cluster_result["dominant_cluster_label"]
                rec["dtw_cluster_share"] = cluster_result["dominant_cluster_share"]
                rec["dtw_best_k"] = cluster_result["fit_summary"]["best_k"]
                rec["dtw_silhouette"] = cluster_result["fit_summary"]["silhouette"]
            elif daily["dtw_cluster_label"].notna().any():
                rec["dtw_cluster_label"] = daily["dtw_cluster_label"].mode(dropna=True).iloc[0]
            else:
                rec["dtw_cluster_label"] = np.nan
            records.append(rec)
        else:
            tmp = daily.copy()
            tmp["fsa"] = code
            records.append(tmp.reset_index())

    if not records:
        return pd.DataFrame()

    if aggregate:
        return pd.DataFrame(records).set_index("fsa").sort_index()

    out = pd.concat(records, ignore_index=True)
    return out.set_index(["fsa", "date"]).sort_index()
