from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _to_numeric_census(census_df: pd.DataFrame) -> pd.DataFrame:
    out = census_df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _monthly_energy_by_fsa(elec_df: pd.DataFrame, agg: str = "sum") -> pd.DataFrame:
    if not isinstance(elec_df.index, pd.DatetimeIndex):
        raise TypeError("elec_df index must be a DatetimeIndex.")
    if agg not in {"sum", "mean"}:
        raise ValueError("agg must be one of: 'sum', 'mean'.")
    m = elec_df.resample("MS").sum() if agg == "sum" else elec_df.resample("MS").mean()
    return m


def select_census_features_for_energy(
    elec_df: pd.DataFrame,
    census_df: pd.DataFrame,
    candidate_features: list[str] | None = None,
    monthly_agg: str = "sum",
    corr_method: str = "spearman",
    min_non_null_ratio: float = 0.8,
    min_unique_values: int = 5,
    max_features: int = 20,
) -> tuple[list[str], pd.DataFrame]:
    """
    Select census predictors by correlation to monthly FSA energy across observed FSAs.

    Returns
    -------
    selected_features : list[str]
    feature_scores : pd.DataFrame
        index=feature, columns=[score_mean_abs_corr, n_months_used, non_null_ratio]
    """
    if corr_method not in {"spearman", "pearson"}:
        raise ValueError("corr_method must be one of: 'spearman', 'pearson'.")

    census_num = _to_numeric_census(census_df)
    if candidate_features is None:
        candidate_features = census_num.columns.tolist()
    else:
        candidate_features = [c for c in candidate_features if c in census_num.columns]
    if not candidate_features:
        raise ValueError("No valid candidate features found in census_df.")

    monthly = _monthly_energy_by_fsa(elec_df, agg=monthly_agg)
    observed_fsas = [c for c in monthly.columns if c in census_num.index]
    if len(observed_fsas) < 5:
        raise ValueError("Too few overlapping FSAs between elec_df columns and census_df index.")

    scores: list[dict] = []
    for f in candidate_features:
        x = census_num.loc[observed_fsas, f]
        nn_ratio = float(x.notna().mean())
        if nn_ratio < min_non_null_ratio:
            continue
        if int(x.nunique(dropna=True)) < min_unique_values:
            continue

        month_corrs = []
        for ts in monthly.index:
            y = monthly.loc[ts, observed_fsas]
            tmp = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
            if len(tmp) < 8:
                continue
            cval = tmp["x"].corr(tmp["y"], method=corr_method)
            if pd.notna(cval):
                month_corrs.append(abs(float(cval)))
        if not month_corrs:
            continue
        scores.append(
            {
                "feature": f,
                "score_mean_abs_corr": float(np.mean(month_corrs)),
                "n_months_used": int(len(month_corrs)),
                "non_null_ratio": nn_ratio,
            }
        )

    if not scores:
        raise ValueError("No feature passed quality filters for correlation-based selection.")

    score_df = pd.DataFrame(scores).set_index("feature").sort_values("score_mean_abs_corr", ascending=False)
    selected = score_df.head(max_features).index.tolist()
    return selected, score_df


def _weighted_standardized_distance(
    donor_mat: pd.DataFrame,
    target_row: pd.Series,
    weights: pd.Series,
) -> pd.Series:
    mu = donor_mat.mean(axis=0)
    sd = donor_mat.std(axis=0, ddof=0).replace(0, np.nan)
    z_d = (donor_mat - mu) / sd
    z_t = (target_row - mu) / sd

    valid_cols = z_t.index[z_t.notna() & weights.reindex(z_t.index).notna()]
    if len(valid_cols) == 0:
        return pd.Series(np.nan, index=donor_mat.index)

    w = weights.reindex(valid_cols).astype(float)
    if float(w.sum()) == 0:
        w = pd.Series(1.0, index=valid_cols)
    w = w / float(w.sum())

    diff2 = (z_d[valid_cols].sub(z_t[valid_cols], axis=1) ** 2).mul(w, axis=1)
    return np.sqrt(diff2.sum(axis=1))


def impute_missing_fsa_energy_by_census_proximity(
    elec_df: pd.DataFrame,
    census_df: pd.DataFrame,
    geometry_fsas: Iterable[str],
    candidate_features: list[str] | None = None,
    max_features: int = 20,
    monthly_agg: str = "sum",
    corr_method: str = "spearman",
    min_non_null_ratio: float = 0.8,
    min_unique_values: int = 5,
    population_col: str = "Population and dwelling counts / Population, 2021",
    apply_population_scaling: bool = True,
    scale_clip: tuple[float, float] = (0.5, 2.0),
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fill missing-geometry FSAs by copying the most similar observed FSA load profile.

    Similarity is computed using selected census features (correlation-ranked against monthly energy).

    Returns
    -------
    elec_filled : pd.DataFrame
        Original elec_df plus imputed missing FSA columns.
    imputation_report : pd.DataFrame
        Per-imputed-FSA donor info and distance.
    feature_scores : pd.DataFrame
        Feature ranking table from selection stage.
    """
    if not isinstance(elec_df.index, pd.DatetimeIndex):
        raise TypeError("elec_df index must be a DatetimeIndex.")
    if not isinstance(census_df.index, pd.Index):
        raise TypeError("census_df must be indexed by FSA code.")

    geom_fsas = sorted({str(x) for x in geometry_fsas})
    observed_fsas = sorted([str(c) for c in elec_df.columns])
    missing_fsas = [f for f in geom_fsas if f not in observed_fsas]
    if len(missing_fsas) == 0:
        return elec_df.copy(), pd.DataFrame(), pd.DataFrame()

    selected_features, feature_scores = select_census_features_for_energy(
        elec_df=elec_df,
        census_df=census_df,
        candidate_features=candidate_features,
        monthly_agg=monthly_agg,
        corr_method=corr_method,
        min_non_null_ratio=min_non_null_ratio,
        min_unique_values=min_unique_values,
        max_features=max_features,
    )

    census_num = _to_numeric_census(census_df)
    donor_pool = [f for f in observed_fsas if f in census_num.index]
    donor_mat = census_num.loc[donor_pool, selected_features]
    donor_mat = donor_mat.dropna(how="all")
    donor_pool = donor_mat.index.astype(str).tolist()
    if len(donor_pool) == 0:
        raise ValueError("No donor FSAs have usable census features.")

    weights = feature_scores.reindex(selected_features)["score_mean_abs_corr"].fillna(0.0)
    if float(weights.sum()) == 0:
        weights[:] = 1.0
    weights = weights / float(weights.sum())

    elec_filled = elec_df.copy()
    records: list[dict] = []

    miss_iter = tqdm(missing_fsas, desc="Imputing missing FSAs") if show_progress else missing_fsas
    for fsa_miss in miss_iter:
        if fsa_miss not in census_num.index:
            records.append(
                {
                    "fsa_missing": fsa_miss,
                    "fsa_donor": np.nan,
                    "distance": np.nan,
                    "scale_factor": np.nan,
                    "status": "missing_census",
                }
            )
            continue

        x_t = census_num.loc[fsa_miss, selected_features]
        dists = _weighted_standardized_distance(donor_mat=donor_mat, target_row=x_t, weights=weights)
        dists = dists.dropna()
        if dists.empty:
            records.append(
                {
                    "fsa_missing": fsa_miss,
                    "fsa_donor": np.nan,
                    "distance": np.nan,
                    "scale_factor": np.nan,
                    "status": "no_comparable_donor",
                }
            )
            continue

        donor = str(dists.idxmin())
        distance = float(dists.loc[donor])
        donor_series = elec_df[donor].copy()

        scale_factor = 1.0
        if apply_population_scaling and population_col in census_num.columns:
            pop_t = pd.to_numeric(census_num.at[fsa_miss, population_col], errors="coerce")
            pop_d = pd.to_numeric(census_num.at[donor, population_col], errors="coerce")
            if pd.notna(pop_t) and pd.notna(pop_d) and float(pop_d) > 0:
                scale_factor = float(pop_t) / float(pop_d)
                lo, hi = float(scale_clip[0]), float(scale_clip[1])
                scale_factor = float(np.clip(scale_factor, lo, hi))

        elec_filled[fsa_miss] = donor_series * scale_factor
        records.append(
            {
                "fsa_missing": fsa_miss,
                "fsa_donor": donor,
                "distance": distance,
                "scale_factor": scale_factor,
                "status": "imputed",
            }
        )

    report = pd.DataFrame(records).set_index("fsa_missing").sort_index()
    return elec_filled, report, feature_scores


def _paired_error_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mape_epsilon: float = 1e-9,
) -> dict[str, float]:
    pair = pd.concat([y_true.rename("true"), y_pred.rename("pred")], axis=1).dropna()
    if pair.empty:
        return {
            "n_obs": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "mbe": np.nan,
            "mape_pct": np.nan,
            "corr": np.nan,
        }

    err = pair["pred"] - pair["true"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mbe = float(np.mean(err))

    denom = np.abs(pair["true"]).astype(float)
    denom = denom.where(denom >= float(mape_epsilon), np.nan)
    mape_pct = float(np.nanmean(np.abs(err) / denom) * 100.0) if denom.notna().any() else np.nan

    corr = np.nan
    if len(pair) >= 2:
        corr = float(pair["true"].corr(pair["pred"]))

    return {
        "n_obs": float(len(pair)),
        "mae": mae,
        "rmse": rmse,
        "mbe": mbe,
        "mape_pct": mape_pct,
        "corr": corr,
    }


def evaluate_imputation_holdout(
    elec_df: pd.DataFrame,
    census_df: pd.DataFrame,
    geometry_fsas: Iterable[str],
    holdout_fsas: list[str] | None = None,
    n_holdout: int = 20,
    random_state: int | None = 42,
    monthly_freq: str = "MS",
    mape_epsilon: float = 1e-9,
    show_progress: bool = True,
    **impute_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate FSA imputation quality with pseudo-missing holdout FSAs.

    The function removes selected observed FSAs, imputes them back using
    census-proximity donor matching, and computes error metrics by holdout FSA.

    Parameters
    ----------
    elec_df : pd.DataFrame
        Wide electricity dataframe (index=time, columns=FSA).
    census_df : pd.DataFrame
        Census table indexed by FSA code.
    geometry_fsas : Iterable[str]
        All geometry FSA codes expected in the city.
    holdout_fsas : list[str] | None
        Explicit holdout FSA codes. If None, sample `n_holdout` from elec_df columns.
    n_holdout : int
        Number of holdout FSAs to sample when holdout_fsas is None.
    random_state : int | None
        Random seed for holdout sampling.
    monthly_freq : str
        Resample frequency for energy-accuracy metrics (default "MS").
    mape_epsilon : float
        Minimum absolute true value allowed in MAPE denominator.
    show_progress : bool
        Show progress bars inside imputation call.
    **impute_kwargs
        Additional arguments forwarded to
        `impute_missing_fsa_energy_by_census_proximity`.

    Returns
    -------
    metrics_df : pd.DataFrame
        Per-holdout metrics including hourly and monthly MAE/RMSE/MBE/MAPE/Corr.
    report : pd.DataFrame
        Imputation donor report for holdouts.
    feature_scores : pd.DataFrame
        Feature ranking table from the imputation routine.
    elec_filled : pd.DataFrame
        Electricity dataframe after re-imputing holdout FSAs.
    """
    if not isinstance(elec_df.index, pd.DatetimeIndex):
        raise TypeError("elec_df index must be a DatetimeIndex.")

    observed_fsas = [str(c) for c in elec_df.columns]
    if holdout_fsas is None:
        if n_holdout <= 0:
            raise ValueError("n_holdout must be > 0 when holdout_fsas is None.")
        n_use = min(int(n_holdout), len(observed_fsas))
        rng = np.random.default_rng(random_state)
        holdout_fsas = sorted(rng.choice(observed_fsas, size=n_use, replace=False).tolist())
    else:
        holdout_fsas = sorted({str(c) for c in holdout_fsas})

    if len(holdout_fsas) == 0:
        raise ValueError("No holdout FSAs provided/found for evaluation.")

    present_holdouts = [c for c in holdout_fsas if c in elec_df.columns]
    if len(present_holdouts) == 0:
        raise ValueError("None of the requested holdout FSAs are present in elec_df columns.")

    elec_train = elec_df.drop(columns=present_holdouts, errors="ignore")
    elec_filled, report, feature_scores = impute_missing_fsa_energy_by_census_proximity(
        elec_df=elec_train,
        census_df=census_df,
        geometry_fsas=geometry_fsas,
        show_progress=show_progress,
        **impute_kwargs,
    )

    rows: list[dict] = []
    for fsa in present_holdouts:
        y_true = elec_df[fsa]
        y_pred = elec_filled[fsa] if fsa in elec_filled.columns else pd.Series(dtype=float)

        hourly = _paired_error_metrics(y_true=y_true, y_pred=y_pred, mape_epsilon=mape_epsilon)

        yt_m = y_true.resample(monthly_freq).sum()
        yp_m = y_pred.resample(monthly_freq).sum() if len(y_pred) else pd.Series(dtype=float)
        monthly = _paired_error_metrics(y_true=yt_m, y_pred=yp_m, mape_epsilon=mape_epsilon)

        rec = {
            "fsa": fsa,
            **{f"hourly_{k}": v for k, v in hourly.items()},
            **{f"monthly_{k}": v for k, v in monthly.items()},
        }
        if isinstance(report, pd.DataFrame) and not report.empty and fsa in report.index:
            rec["donor_fsa"] = report.at[fsa, "fsa_donor"] if "fsa_donor" in report.columns else np.nan
            rec["donor_distance"] = report.at[fsa, "distance"] if "distance" in report.columns else np.nan
            rec["scale_factor"] = report.at[fsa, "scale_factor"] if "scale_factor" in report.columns else np.nan
            rec["status"] = report.at[fsa, "status"] if "status" in report.columns else np.nan
        rows.append(rec)

    metrics_df = pd.DataFrame(rows).set_index("fsa").sort_index()
    return metrics_df, report, feature_scores, elec_filled
