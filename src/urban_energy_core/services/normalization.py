from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(census_row: pd.Series | dict | None) -> pd.Series:
    if census_row is None:
        raise ValueError("census_row is required for per-capita calculations.")
    if isinstance(census_row, pd.Series):
        return census_row
    return pd.Series(census_row)


def align_weather_to_load(
    load_series: pd.Series,
    weather_df: pd.DataFrame,
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
) -> pd.DataFrame:
    def _localize_with_dst_handling(idx: pd.DatetimeIndex, tz) -> pd.DatetimeIndex:
        try:
            return idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            return idx.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")

    if temp_col not in weather_df.columns:
        raise KeyError(f"Temperature column '{temp_col}' not found in weather dataframe.")
    if dt_col in weather_df.columns:
        wx = weather_df[[dt_col, temp_col]].copy()
        wx[dt_col] = pd.to_datetime(wx[dt_col], errors="coerce")
        wx = wx.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    elif isinstance(weather_df.index, pd.DatetimeIndex):
        wx = weather_df[[temp_col]].copy().sort_index()
    else:
        raise KeyError(f"Datetime column '{dt_col}' not found and weather index is not DatetimeIndex.")

    y = load_series.astype(float).rename("load").to_frame()
    y.index = pd.to_datetime(y.index, errors="coerce")
    y = y.dropna().sort_index()

    # Harmonize tz-awareness between load index and weather index.
    y_tz = y.index.tz
    wx_tz = wx.index.tz
    if y_tz is not None and wx_tz is None:
        wx.index = _localize_with_dst_handling(wx.index, y_tz)
        wx = wx[~wx.index.isna()]
    elif y_tz is None and wx_tz is not None:
        wx.index = wx.index.tz_localize(None)
    elif y_tz is not None and wx_tz is not None and str(y_tz) != str(wx_tz):
        wx.index = wx.index.tz_convert(y_tz)

    joined = y.join(wx[[temp_col]], how="left")
    joined[temp_col] = joined[temp_col].interpolate(limit_direction="both")
    return joined.dropna(subset=[temp_col])


def normalize_fsa_weather_linear(
    load_series: pd.Series,
    weather_df: pd.DataFrame,
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
) -> pd.Series:
    """
    Simple weather normalization:
    load_norm = load - beta * (temp - temp_mean)
    """
    joined = align_weather_to_load(
        load_series=load_series,
        weather_df=weather_df,
        dt_col=dt_col,
        temp_col=temp_col,
    )
    x = joined[temp_col].astype(float).to_numpy()
    y = joined["load"].astype(float).to_numpy()

    x_mean = float(np.mean(x))
    var = float(np.var(x))
    beta = 0.0 if var == 0 else float(np.cov(x, y, ddof=0)[0, 1] / var)
    y_norm = y - beta * (x - x_mean)

    out = pd.Series(y_norm, index=joined.index, name=load_series.name)
    out = out.reindex(pd.to_datetime(load_series.index, errors="coerce"))
    return out


def compute_per_capita_series(
    load_series: pd.Series,
    census_row: pd.Series | dict | None,
    population_col: str = "Population and dwelling counts / Population, 2021",
) -> pd.Series:
    row = _to_series(census_row)
    if population_col not in row.index:
        raise KeyError(f"Population column '{population_col}' not found in census row.")
    pop = float(pd.to_numeric(row[population_col], errors="coerce"))
    if not np.isfinite(pop) or pop <= 0:
        raise ValueError(f"Invalid population value: {row[population_col]}")
    out = load_series.astype(float) / pop
    out.name = f"{load_series.name}_per_capita" if load_series.name else "per_capita"
    return out
