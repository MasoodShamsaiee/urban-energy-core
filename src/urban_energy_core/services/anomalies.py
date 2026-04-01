import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from tqdm.auto import tqdm

def hampel_filter_series(s: pd.Series, window: int = 24, n_sigmas: float = 5.0, replace: str = "interp"):
    """
    Hampel filter for a single time series.
    window: rolling window size in number of rows (e.g., 24 for hourly ~1 day)
    n_sigmas: threshold in robust sigmas
    replace: "interp" or "median"
    """
    x = s.astype(float).copy()

    med = x.rolling(window, center=True, min_periods=max(3, window//3)).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=max(3, window//3)).median()

    # Robust sigma estimate
    sigma = 1.4826 * mad
    mask = (sigma > 0) & ((x - med).abs() > n_sigmas * sigma)

    x_clean = x.copy()
    if replace == "median":
        x_clean[mask] = med[mask]
    else:
        x_clean[mask] = np.nan
        x_clean = x_clean.interpolate(limit_direction="both")

    return x_clean, mask

def clean_spikes_hampel(
    elec_df: pd.DataFrame,
    fsas=None,
    window: int = 24,
    n_sigmas: float = 5.0,
    replace="interp",
    show_progress: bool = True,
):
    """
    Apply Hampel filter to selected FSAs (columns). Returns (clean_df, mask_df).
    """
    df = elec_df.copy()
    if fsas is None:
        fsas = df.columns.tolist()
    elif isinstance(fsas, str):
        fsas = [fsas]
    else:
        fsas = list(fsas)

    mask_df = pd.DataFrame(False, index=df.index, columns=fsas)
    fsa_iter = tqdm(fsas, desc="Hampel cleaning FSAs") if show_progress else fsas
    for fsa in fsa_iter:
        cleaned, mask = hampel_filter_series(df[fsa], window=window, n_sigmas=n_sigmas, replace=replace)
        df[fsa] = cleaned
        mask_df[fsa] = mask.fillna(False)

    return df, mask_df


def stl_anomaly_analysis(
    s: pd.Series,
    period: int,
    z_thresh: float = 3.5,
    robust: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect anomalies with STL decomposition and robust residual scoring.

    Parameters
    ----------
    s : pd.Series
        Input time series.
    period : int
        Seasonal period in rows (e.g., 24 for hourly daily seasonality).
    z_thresh : float, default 3.5
        Absolute robust z-score threshold to flag anomalies.
    robust : bool, default True
        Whether to use robust STL fitting.

    Returns
    -------
    components : pd.DataFrame
        Columns: value, trend, seasonal, resid, robust_z, is_anomaly, anomaly_direction.
    summary : dict
        Compact anomaly analysis summary.
    """
    if period < 2:
        raise ValueError("`period` must be >= 2.")
    if len(s) < 2 * period:
        raise ValueError("Series is too short for STL. Provide at least 2 * period points.")

    x = s.astype(float)
    x_fit = x.interpolate(limit_direction="both")

    stl = STL(x_fit, period=period, robust=robust).fit()
    resid = stl.resid

    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if not np.isfinite(mad) or mad == 0:
        std = np.nanstd(resid)
        robust_z = np.zeros_like(resid, dtype=float) if std == 0 else (resid - med) / std
    else:
        robust_z = 0.6745 * (resid - med) / mad

    is_anomaly = np.abs(robust_z) >= z_thresh
    anomaly_direction = np.where(
        is_anomaly & (resid >= 0),
        "high",
        np.where(is_anomaly, "low", "normal"),
    )

    components = pd.DataFrame(
        {
            "value": x,
            "trend": stl.trend,
            "seasonal": stl.seasonal,
            "resid": resid,
            "robust_z": robust_z,
            "is_anomaly": is_anomaly,
            "anomaly_direction": anomaly_direction,
        },
        index=s.index,
    )

    n_anom = int(components["is_anomaly"].sum())
    summary = {
        "period": int(period),
        "z_threshold": float(z_thresh),
        "n_observations": int(len(components)),
        "n_missing_input": int(s.isna().sum()),
        "n_anomalies": n_anom,
        "anomaly_rate": float(n_anom / len(components)),
        "n_high_anomalies": int((components["anomaly_direction"] == "high").sum()),
        "n_low_anomalies": int((components["anomaly_direction"] == "low").sum()),
        "mean_abs_resid": float(np.nanmean(np.abs(components["resid"]))),
        "max_abs_resid": float(np.nanmax(np.abs(components["resid"]))),
    }

    return components, summary


def replace_stl_anomalies(
    components: pd.DataFrame,
    method: str = "interp",
    rolling_window: int = 24,
) -> tuple[pd.Series, pd.Series]:
    """
    Replace anomalies detected by `stl_anomaly_analysis`.

    Parameters
    ----------
    components : pd.DataFrame
        Dataframe returned by `stl_anomaly_analysis`.
    method : str, default "interp"
        Replacement strategy:
        - "interp": set anomalies to NaN then time-interpolate.
        - "stl_rebuild": replace with trend + seasonal.
        - "rolling_median": replace with centered rolling median of `value`.
    rolling_window : int, default 24
        Window for "rolling_median" method.

    Returns
    -------
    cleaned : pd.Series
        Series after anomaly replacement.
    replaced_mask : pd.Series
        Boolean mask of replaced points.
    """
    required = {"value", "trend", "seasonal", "is_anomaly"}
    missing = required.difference(components.columns)
    if missing:
        raise KeyError(f"Missing required columns in components: {sorted(missing)}")

    replaced_mask = components["is_anomaly"].fillna(False).astype(bool)
    base = components["value"].astype(float).copy()

    if method == "interp":
        cleaned = base.copy()
        cleaned[replaced_mask] = np.nan
        cleaned = cleaned.interpolate(limit_direction="both")
    elif method == "stl_rebuild":
        rebuilt = (components["trend"] + components["seasonal"]).astype(float)
        cleaned = base.where(~replaced_mask, rebuilt)
    elif method == "rolling_median":
        if rolling_window < 3:
            raise ValueError("`rolling_window` must be >= 3 for rolling_median.")
        med = base.rolling(
            window=rolling_window,
            center=True,
            min_periods=max(3, rolling_window // 3),
        ).median()
        cleaned = base.where(~replaced_mask, med)
        cleaned = cleaned.interpolate(limit_direction="both")
    else:
        raise ValueError("Invalid method. Use one of: 'interp', 'stl_rebuild', 'rolling_median'.")

    cleaned.name = f"{base.name}_clean" if base.name is not None else "cleaned"
    replaced_mask.name = "is_replaced"
    return cleaned, replaced_mask


def treat_anomalies_until_target_rate(
    elec_df: pd.DataFrame,
    target_rate: float = 0.001,
    period: int = 24,
    z_thresh: float = 3.0,
    robust: bool = False,
    max_iter: int = 10,
    method: str = "interp",
    fsas: list[str] | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Iteratively detect/replace STL anomalies per FSA until target anomaly rate.

    Parameters
    ----------
    elec_df : pd.DataFrame
        Wide electricity dataframe (index=time, columns=FSA).
    target_rate : float, default 0.001
        Stop when anomaly_rate <= target_rate.
    period : int, default 24
        STL seasonal period.
    z_thresh : float, default 3.0
        Robust z-score threshold for anomaly flagging.
    robust : bool, default False
        Passed to STL robust fitting.
    max_iter : int, default 10
        Safety cap on per-FSA iterations.
    method : str, default "interp"
        Replacement method passed to `replace_stl_anomalies`.
    fsas : list[str] | None
        Optional subset of columns to process. If None, process all columns.

    Returns
    -------
    elec_clean_anomaly_treated : pd.DataFrame
        Treated dataframe.
    components_by_fsa : dict[str, pd.DataFrame]
        Final STL components for each processed FSA.
    conformance_report : pd.DataFrame
        Per-FSA status summary.
    """
    if not (0 <= target_rate <= 1):
        raise ValueError("`target_rate` must be in [0, 1].")
    if max_iter < 1:
        raise ValueError("`max_iter` must be >= 1.")

    out_df = elec_df.copy()
    process_fsas = out_df.columns.tolist() if fsas is None else list(fsas)

    missing = [f for f in process_fsas if f not in out_df.columns]
    if missing:
        raise KeyError(f"FSAs not found in dataframe: {missing}")

    records = []
    components_by_fsa: dict[str, pd.DataFrame] = {}

    outer_iter = tqdm(process_fsas, desc="Treating anomalies by FSA") if show_progress else process_fsas
    for fsa in outer_iter:
        s_current = out_df[fsa].copy()
        final_components = None
        final_summary = None
        replaced_total = 0
        status = "ok"

        iter_range = range(1, max_iter + 1)
        inner_iter = (
            tqdm(iter_range, desc=f"{fsa} iterations", leave=False)
            if show_progress
            else iter_range
        )
        for i in inner_iter:
            components, summary = stl_anomaly_analysis(
                s=s_current,
                period=period,
                z_thresh=z_thresh,
                robust=robust,
            )
            final_components = components
            final_summary = summary

            if summary["anomaly_rate"] <= target_rate:
                break

            s_next, replaced_mask = replace_stl_anomalies(components, method=method)
            replaced_total += int(replaced_mask.sum())

            if s_next.equals(s_current):
                status = "stalled"
                break

            s_current = s_next

        if final_summary is None or final_components is None:
            raise RuntimeError(f"Unexpected empty STL results for FSA '{fsa}'.")

        if final_summary["anomaly_rate"] > target_rate and status == "ok":
            status = "not_conformed"

        out_df[fsa] = s_current
        components_by_fsa[fsa] = final_components

        records.append(
            {
                "fsa": fsa,
                "final_anomaly_rate": final_summary["anomaly_rate"],
                "final_n_anomalies": final_summary["n_anomalies"],
                "iterations": i,
                "replaced_points_total": replaced_total,
                "status": status,
            }
        )

    conformance_report = (
        pd.DataFrame(records)
        .set_index("fsa")
        .sort_values("final_anomaly_rate", ascending=False)
    )

    return out_df, components_by_fsa, conformance_report
