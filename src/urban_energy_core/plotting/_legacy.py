import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import json

from urban_energy_core.services.prism import fit_prism_segmented, predict_prism_segmented


def _normalize_city_fsa_selection(city, fsas=None) -> list[str]:
    if fsas is None:
        return city.list_fsa_codes()

    if isinstance(fsas, str):
        raw = [fsas]
    elif hasattr(fsas, "code"):
        raw = [fsas.code]
    else:
        try:
            raw = []
            for item in fsas:
                if isinstance(item, str):
                    raw.append(item)
                elif hasattr(item, "code"):
                    raw.append(item.code)
                else:
                    raise TypeError(
                        "Each FSA selector must be a code string or an FSA-like object with a 'code' attribute."
                    )
        except TypeError as exc:
            raise TypeError(
                "fsas must be None, an FSA code string, an FSA object, or an iterable of those."
            ) from exc

    codes = []
    seen = set()
    for c in raw:
        code = str(c)
        if code in seen:
            continue
        seen.add(code)
        if code not in city.fsas:
            raise KeyError(f"FSA '{code}' not found in city '{city.name}'.")
        codes.append(code)
    return codes


def plot_city_fsa_map(
    city,
    fsas=None,
    *,
    metric: str = "mean",
    alpha: float = 0.8,
    figsize: tuple[float, float] = (8, 5),
    start=None,
    end=None,
    title: str | None = None,
    show: bool = True,
):
    """
    Plot a static choropleth map from a City object.

    Parameters
    ----------
    city : src.domain.city.City
        City object containing FSAs with geometry and electricity series.
    fsas : str | FSA | list[str | FSA] | None
        FSA selection by code and/or FSA object(s). If None, plots all FSAs.
    metric : str
        One of: "mean", "sum", "latest", "min", "max".
    alpha : float
        Fill opacity for FSA polygons in [0, 1].
    figsize : tuple[float, float]
        Figure size as (width, height), in inch-like units converted to pixels.
    start, end : str or pd.Timestamp, optional
        Optional date range filter for electricity aggregation.
    title : str or None
        Figure title.
    show : bool
        If True, calls fig.show().
    """
    import geopandas as gpd

    valid_metrics = {"mean", "sum", "latest", "min", "max"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {sorted(valid_metrics)}.")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if len(figsize) != 2 or float(figsize[0]) <= 0 or float(figsize[1]) <= 0:
        raise ValueError("figsize must be a tuple of two positive numbers, e.g. (8, 5).")

    codes = _normalize_city_fsa_selection(city, fsas)
    if not codes:
        raise ValueError("No FSAs selected for plotting.")

    rows = []
    missing_geometry = []
    for code in codes:
        geom = city.fsas[code].geometry
        if geom is None:
            missing_geometry.append(code)
            continue
        rows.append({"FSA": code, "geometry": geom})

    if not rows:
        raise ValueError("No selected FSAs have geometry available for mapping.")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")

    elec_df = city.electricity_frame()
    value_col = "value"
    value_label = "selection"
    if not elec_df.empty:
        use_cols = [c for c in codes if c in elec_df.columns]
        if use_cols:
            s_df = elec_df[use_cols].copy()
            s_df.index = pd.to_datetime(s_df.index, errors="coerce")
            s_df = s_df.dropna(how="all").sort_index()
            if start is not None or end is not None:
                s_df = s_df.loc[start:end]

            if not s_df.empty:
                if metric == "latest":
                    values = s_df.ffill().iloc[-1]
                    value_label = "latest kWh"
                elif metric == "mean":
                    values = s_df.mean(axis=0)
                    value_label = "mean kWh"
                elif metric == "sum":
                    values = s_df.sum(axis=0)
                    value_label = "sum kWh"
                elif metric == "min":
                    values = s_df.min(axis=0)
                    value_label = "min kWh"
                else:
                    values = s_df.max(axis=0)
                    value_label = "max kWh"

                gdf = gdf.merge(values.rename(value_col), left_on="FSA", right_index=True, how="left")

    if value_col not in gdf.columns:
        gdf[value_col] = 1.0

    geojson = json.loads(gdf.to_json())
    centroid = gdf.geometry.union_all().centroid if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union.centroid
    center = {"lat": float(centroid.y), "lon": float(centroid.x)}

    bounds = gdf.total_bounds
    span = max(float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1]))
    if span <= 0:
        zoom = 10.0
    else:
        zoom = float(np.clip(np.log2(360.0 / span) - 1.0, 3.0, 12.0))

    z = pd.to_numeric(gdf[value_col], errors="coerce").to_numpy(dtype=float)
    has_numeric_values = np.isfinite(z).any() and not np.allclose(np.nanmin(z), np.nanmax(z), equal_nan=True)
    if has_numeric_values:
        trace = go.Choroplethmap(
            geojson=geojson,
            locations=gdf["FSA"].astype(str),
            z=z,
            featureidkey="properties.FSA",
            colorscale="Turbo",
            marker_opacity=float(alpha),
            marker_line_width=0.4,
            colorbar={"title": value_label},
            customdata=gdf["FSA"].astype(str).to_numpy(),
            hovertemplate="FSA: %{customdata}<br>"+value_label+": %{z:.2f}<extra></extra>",
        )
    else:
        trace = go.Choroplethmap(
            geojson=geojson,
            locations=gdf["FSA"].astype(str),
            z=np.ones(len(gdf), dtype=float),
            featureidkey="properties.FSA",
            colorscale=[[0, "#2A9D8F"], [1, "#2A9D8F"]],
            showscale=False,
            marker_opacity=float(alpha),
            marker_line_width=0.6,
            customdata=gdf["FSA"].astype(str).to_numpy(),
            hovertemplate="FSA: %{customdata}<extra></extra>",
        )

    if title is None:
        title = f"{city.name}: FSA map ({metric})"
    if missing_geometry:
        title += f" | missing geometry: {len(missing_geometry)}"

    fig = go.Figure(data=[trace])
    width, height = int(float(figsize[0]) * 100), int(float(figsize[1]) * 100)
    fig.update_layout(
        title=title,
        map=dict(style="carto-positron", center=center, zoom=zoom),
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        width=width,
        height=height,
    )

    if show:
        fig.show()
    return fig

def plot_fsa_timeseries(
    elec_df,
    fsas=None,
    *,
    start=None,
    end=None,
    max_fsas=6,
    figsize=(12, 4),
):
    """
    Plot electricity time series for one or more FSAs.

    Parameters
    ----------
    elec_df : pd.DataFrame
        Wide dataframe (index = DatetimeIndex, columns = FSAs)
    fsas : str or list[str] or None
        FSA(s) to plot. If None, plot the first `max_fsas` columns.
    start, end : str or pd.Timestamp, optional
        Time range to plot.
    max_fsas : int
        Max FSAs to plot if fsas is None.
    figsize : tuple
        Figure size.
    """
    import pandas as pd

    # --- sanity checks ---
    if not isinstance(elec_df.index, pd.DatetimeIndex):
        raise TypeError("elec_df index must be a DatetimeIndex")

    if fsas is None:
        fsas = elec_df.columns[:max_fsas].tolist()
    elif isinstance(fsas, str):
        fsas = [fsas]
    else:
        fsas = list(fsas)

    missing = [f for f in fsas if f not in elec_df.columns]
    if missing:
        raise KeyError(f"FSAs not found in dataframe: {missing}")

    df = elec_df[fsas]

    if start or end:
        df = df.loc[start:end]

    if df.empty:
        raise ValueError("No data to plot after applying filters")

    # --- plot ---
    fig = go.Figure()
    for fsa in fsas:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[fsa],
                mode="lines",
                name=fsa,
            )
        )

    width, height = int(figsize[0] * 100), int(figsize[1] * 100)
    fig.update_layout(
        title="Electricity time series by FSA",
        xaxis_title="Time",
        yaxis_title="Electricity consumption (kWh)",
        width=width,
        height=height,
        template="plotly_white",
    )
    fig.show()


def plot_stl_anomalies(
    components,
    *,
    title="STL Anomaly Detection",
    z_thresh=None,
    show_decomposition=True,
    figsize=(13, 8),
):
    """
    Plot STL anomaly detection results from `stl_anomaly_analysis`.

    Parameters
    ----------
    components : pd.DataFrame
        Output dataframe from `stl_anomaly_analysis` with columns:
        value, trend, seasonal, resid, robust_z, is_anomaly, anomaly_direction.
    title : str
        Figure title.
    z_thresh : float or None
        If provided, draws +/- threshold lines on robust_z panel.
    show_decomposition : bool
        If True, show trend/seasonal/residual panels; otherwise only main series.
    figsize : tuple
        Figure size in inches-like units (converted to plotly pixels).
    """
    import pandas as pd

    required = {
        "value",
        "trend",
        "seasonal",
        "resid",
        "robust_z",
        "is_anomaly",
        "anomaly_direction",
    }
    missing = required.difference(components.columns)
    if missing:
        raise KeyError(f"Missing required columns in components: {sorted(missing)}")
    if not isinstance(components.index, pd.DatetimeIndex):
        raise TypeError("components index must be a DatetimeIndex")

    df = components.copy()
    normal = df[~df["is_anomaly"]]
    high = df[df["anomaly_direction"] == "high"]
    low = df[df["anomaly_direction"] == "low"]

    if show_decomposition:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=("Observed with Anomalies", "Trend", "Seasonal", "Robust Z-score"),
            row_heights=[0.5, 0.2, 0.15, 0.15],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # Main panel: normal signal + anomalies highlighted.
    fig.add_trace(
        go.Scatter(
            x=normal.index,
            y=normal["value"],
            mode="lines",
            name="Non-anomaly",
            line=dict(color="#4C78A8", width=1.8),
            opacity=0.85,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=high.index,
            y=high["value"],
            mode="markers",
            name="High anomaly",
            marker=dict(color="#E45756", size=8, symbol="circle"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=low.index,
            y=low["value"],
            mode="markers",
            name="Low anomaly",
            marker=dict(color="#72B7B2", size=8, symbol="diamond"),
        ),
        row=1,
        col=1,
    )

    if show_decomposition:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["trend"],
                mode="lines",
                name="Trend",
                line=dict(color="#F58518", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["seasonal"],
                mode="lines",
                name="Seasonal",
                line=dict(color="#54A24B", width=1.2),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["robust_z"],
                mode="lines",
                name="Robust z",
                line=dict(color="#9D755D", width=1.2),
            ),
            row=4,
            col=1,
        )
        if z_thresh is not None:
            for level in (float(z_thresh), -float(z_thresh)):
                fig.add_hline(
                    y=level,
                    line=dict(color="#E45756", dash="dash", width=1),
                    row=4,
                    col=1,
                )
        fig.update_yaxes(title_text="kWh", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="z", row=4, col=1)
    else:
        fig.update_yaxes(title_text="kWh", row=1, col=1)

    width, height = int(figsize[0] * 100), int(figsize[1] * 100)
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.show()


def visualize_census_column_hierarchy_zoomable(
    census_df,
    split_pattern=r"\s*/\s*",
    chart="icicle",
    height=900,
):
    """
    Build a zoomable hierarchy from census dataframe column names.

    Column names are split into hierarchical levels using `split_pattern`.
    Supported chart types: "icicle", "sunburst".
    """
    if chart not in {"icicle", "sunburst"}:
        raise ValueError("chart must be 'icicle' or 'sunburst'")

    counts = defaultdict(int)
    labels = {}

    for col in census_df.columns:
        parts = [p.strip() for p in re.split(split_pattern, str(col)) if p.strip()]
        if not parts:
            continue
        for i in range(len(parts)):
            node_id = " / ".join(parts[: i + 1])
            counts[node_id] += 1
            labels[node_id] = parts[i]

    if not counts:
        raise ValueError("No valid hierarchy parsed from columns.")

    ids = []
    parents = []
    node_labels = []
    values = []

    for node_id, val in counts.items():
        parent_id = node_id.rsplit(" / ", 1)[0] if " / " in node_id else ""
        ids.append(node_id)
        parents.append(parent_id)
        node_labels.append(labels[node_id])
        values.append(val)

    if chart == "sunburst":
        fig = go.Figure(
            go.Sunburst(
                ids=ids,
                parents=parents,
                labels=node_labels,
                values=values,
                branchvalues="total",
            )
        )
    else:
        fig = go.Figure(
            go.Icicle(
                ids=ids,
                parents=parents,
                labels=node_labels,
                values=values,
                branchvalues="total",
            )
        )

    fig.update_layout(height=height, margin=dict(t=40, l=10, r=10, b=10))
    fig.show()
    return fig


def plot_city_prism_scatter(
    prism_df: pd.DataFrame,
    x_col: str = "baseload_intercept",
    y_col: str = "heating_slope_per_hdd",
    color_col: str = "heating_change_point_temp_c",
    label_col: str | None = None,
    title: str = "City PRISM: Baseload vs Heating Slope",
    height: int = 700,
):
    """
    Scatter plot for city-level FSA PRISM summaries.
    """
    required = {x_col, y_col, color_col}
    missing = required.difference(prism_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in prism_df: {sorted(missing)}")

    df = prism_df.copy()
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    c = pd.to_numeric(df[color_col], errors="coerce")

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text" if label_col is not None else "markers",
            text=df[label_col].astype(str) if label_col is not None else None,
            textposition="top center",
            marker=dict(
                size=10,
                color=c,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_col),
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=(
                "FSA: %{customdata}<br>"
                f"{x_col}: "+"%{x:.3f}<br>"
                f"{y_col}: "+"%{y:.3f}<br>"
                f"{color_col}: "+"%{marker.color:.2f}<extra></extra>"
            ),
            customdata=df.index.astype(str),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=height,
    )
    fig.show()
    return fig


def plot_fsa_prism_fit(
    fsa,
    weather_df: pd.DataFrame,
    *,
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
    models: tuple[str, ...] = ("2ch", "2cl", "3seg"),
    sample_size: int | None = 10000,
    random_state: int = 42,
    point_alpha: float = 0.35,
    point_size: int = 5,
    show_confidence_band: bool = True,
    height: int = 650,
    title: str | None = None,
    show: bool = True,
):
    """
    Scatter plot of hourly load vs outdoor temperature for one FSA with PRISM fitted line(s).

    Parameters
    ----------
    fsa : src.domain.fsa.FSA
        FSA object containing an electricity series.
    weather_df : pd.DataFrame
        Weather table used to align outdoor temperature to load timestamps.
    models : tuple[str, ...]
        Candidate segmented models to evaluate. Supported: "2ch", "2cl", "3seg".
    """
    if not hasattr(fsa, "electricity") or fsa.electricity is None:
        raise ValueError("fsa must have a non-null electricity series.")
    if not (0.0 <= float(point_alpha) <= 1.0):
        raise ValueError("point_alpha must be in [0, 1].")
    if int(point_size) <= 0:
        raise ValueError("point_size must be > 0.")

    fit = fit_prism_segmented(
        load_series=fsa.electricity,
        weather_df=weather_df,
        dt_col=dt_col,
        temp_col=temp_col,
        models=models,
    )
    joined = fit["joined"][[temp_col, "load"]].dropna()
    if joined.empty:
        raise ValueError("No aligned weather/load points available for plotting.")

    plot_df = joined
    if sample_size is not None and len(plot_df) > int(sample_size):
        plot_df = plot_df.sample(n=int(sample_size), random_state=int(random_state)).sort_index()

    x_all = joined[temp_col].astype(float).to_numpy()
    x_grid = np.linspace(float(np.nanmin(x_all)), float(np.nanmax(x_all)), 300)
    y_fit = predict_prism_segmented(x_grid, fit)
    fit_name = f"{fit['model']} fit (R2={fit['r2']:.3f}, CVRMSE={fit['cvrmse']:.3f})"
    cp_left = float(fit["x1"])
    cp_right = float(fit["x2"])

    resid = np.asarray(fit["residuals"], dtype=float)
    sigma = float(np.nanstd(resid, ddof=1)) if len(resid) > 2 else float("nan")

    fig = go.Figure()
    hover_time = pd.to_datetime(plot_df.index, errors="coerce")
    hover_time_str = hover_time.strftime("%Y-%m-%d %H:%M")
    hover_time_str = np.where(pd.isna(hover_time), "N/A", hover_time_str)
    fig.add_trace(
        go.Scatter(
            x=plot_df[temp_col],
            y=plot_df["load"],
            mode="markers",
            name="Hourly observations",
            marker=dict(size=int(point_size), opacity=float(point_alpha), color="#4C78A8"),
            hovertemplate=(
                "time=%{customdata}<br>"
                "Temp=%{x:.2f} C<br>"
                "Load=%{y:.2f}<extra></extra>"
            ),
            customdata=hover_time_str,
        )
    )
    if show_confidence_band and np.isfinite(sigma):
        y_low = y_fit - 1.96 * sigma
        y_high = y_fit + 1.96 * sigma
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_high,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_low,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(228,87,86,0.12)",
                name="Approx. 95% band",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_fit,
            mode="lines",
            name=fit_name,
            line=dict(color="#E45756", width=3),
            hovertemplate="Temp=%{x:.2f} C<br>Fitted load=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=cp_left,
        line=dict(color="#6C757D", dash="dash", width=1.5),
        annotation_text=f"x1: {cp_left:.1f} C",
        annotation_position="top right",
    )
    if fit["model"] in {"3seg", "3sg", "4sg", "3be"}:
        fig.add_vline(
            x=cp_right,
            line=dict(color="#6C757D", dash="dash", width=1.5),
            annotation_text=f"x2: {cp_right:.1f} C",
            annotation_position="top left",
        )

    fsa_code = getattr(fsa, "code", "FSA")
    if title is None:
        title = f"PRISM Segmented Fit: {fsa_code} ({fit['model']})"

    fig.update_layout(
        title=title,
        xaxis_title="Outdoor temperature (C)",
        yaxis_title="Electricity consumption",
        template="plotly_white",
        height=int(height),
    )

    if show:
        fig.show()
    return fig


def plot_dtw_cluster_bands(
    cluster_result: dict,
    clusters: list[str] | None = None,
    cluster_color_map: dict[str, str] | None = None,
    title: str = "DTW Daily Profile Clusters (P50 with 90% band)",
    height: int = 650,
):
    """
    Plot cluster centroid and 90% band from `cluster_daily_profiles_dtw` output.
    Band is [p05, p95] and centroid is p50.
    """
    if "cluster_hourly_stats" not in cluster_result:
        raise KeyError("cluster_result missing 'cluster_hourly_stats'.")

    stats = cluster_result["cluster_hourly_stats"]
    if not isinstance(stats, pd.DataFrame) or stats.empty:
        raise ValueError("No cluster hourly stats available to plot.")

    required = {"cluster_label", "hour", "p05", "p50", "p95", "n_days"}
    missing = required.difference(stats.columns)
    if missing:
        raise KeyError(f"cluster_hourly_stats missing columns: {sorted(missing)}")

    plot_df = stats.copy().sort_values(["cluster_label", "hour"])
    all_clusters = plot_df["cluster_label"].dropna().unique().tolist()
    use_clusters = all_clusters if clusters is None else [c for c in clusters if c in all_clusters]
    if not use_clusters:
        raise ValueError("No matching clusters to plot.")

    default_cluster_colors = {
        "cluster_0": "#ff7f0e",  # matplotlib tab:orange
        "cluster_1": "#1f77b4",  # matplotlib tab:blue
    }
    fallback_colors = [
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_overrides = default_cluster_colors.copy()
    if cluster_color_map:
        color_overrides.update(cluster_color_map)

    fig = go.Figure()
    fallback_i = 0
    for i, c in enumerate(use_clusters):
        cdf = plot_df[plot_df["cluster_label"] == c]
        color = color_overrides.get(c)
        if color is None:
            color = fallback_colors[fallback_i % len(fallback_colors)]
            fallback_i += 1
        n_days = int(cdf["n_days"].iloc[0]) if len(cdf) else 0

        fig.add_trace(
            go.Scatter(
                x=cdf["hour"],
                y=cdf["p95"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cdf["hour"],
                y=cdf["p05"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
                name=f"{c} 90% band",
                legendgroup=c,
                hovertemplate="hour=%{x}<br>p05=%{y:.3f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cdf["hour"],
                y=cdf["p50"],
                mode="lines",
                line=dict(color=color, width=2.2),
                name=f"{c} p50 (n={n_days})",
                legendgroup=c,
                hovertemplate="hour=%{x}<br>p50=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hour of day",
        yaxis_title="Load profile (scaled if normalization was used)",
        template="plotly_white",
        height=height,
    )
    fig.show()
    return fig


def plot_dtw_dominant_cluster_band(
    cluster_result: dict,
    height: int = 550,
):
    """
    Plot only the dominant DTW cluster (p50 with 90% band).
    Raises if there is no dominant decision.
    """
    if "dominant_cluster_label" not in cluster_result:
        raise KeyError("cluster_result missing 'dominant_cluster_label'.")
    if "dominant_cluster_share" not in cluster_result:
        raise KeyError("cluster_result missing 'dominant_cluster_share'.")

    dom_label = cluster_result["dominant_cluster_label"]
    dom_share = cluster_result["dominant_cluster_share"]

    if dom_label in (None, "no_decision"):
        raise ValueError("No dominant DTW cluster decision. Lower threshold or inspect all clusters.")

    pct = float(dom_share) * 100 if pd.notna(dom_share) else float("nan")
    title = f"Dominant DTW Cluster: {dom_label} (share={pct:.1f}%)"
    return plot_dtw_cluster_bands(
        cluster_result=cluster_result,
        clusters=[dom_label],
        title=title,
        height=height,
    )


def plot_dtw_daily_label_timeline(
    cluster_result: dict,
    title: str = "Daily DTW Cluster Labels",
    height: int = 450,
):
    """
    Plot day-by-day DTW labels (x=date, y=cluster label).
    """
    if "daily_labels" not in cluster_result:
        raise KeyError("cluster_result missing 'daily_labels'.")

    labels = cluster_result["daily_labels"]
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValueError("No daily labels available to plot.")

    s = labels.astype("string").copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    if s.empty:
        raise ValueError("No valid daily labels available to plot.")

    y_order = sorted(s.dropna().unique().tolist())
    y_map = {lab: i for i, lab in enumerate(y_order)}
    y_num = s.map(y_map).astype(float)

    fig = go.Figure(
        go.Scatter(
            x=s.index,
            y=y_num,
            mode="markers",
            marker=dict(size=8, color=y_num, colorscale="Turbo", showscale=False),
            text=s.values,
            hovertemplate="date=%{x|%Y-%m-%d}<br>label=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Day",
        yaxis_title="DTW label",
        template="plotly_white",
        height=height,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(y_map.values()),
        ticktext=list(y_map.keys()),
    )
    fig.show()
    return fig


def plot_dtw_label_distribution_calendar(
    cluster_result: dict,
    normalize: bool = True,
    title: str = "DTW Label Distribution by Calendar Groups",
    height: int = 950,
):
    """
    Show DTW label distributions by:
    - weekday/weekend
    - season
    - month
    """
    if "daily_labels" not in cluster_result:
        raise KeyError("cluster_result missing 'daily_labels'.")

    labels = cluster_result["daily_labels"]
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValueError("No daily labels available to plot.")

    s = labels.astype("string").copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    if s.empty:
        raise ValueError("No valid daily labels available to plot.")

    df = pd.DataFrame({"label": s.values}, index=s.index)
    df["weekpart"] = df.index.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")
    month = df.index.month
    df["month"] = month.map(
        {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
    )
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df["season"] = month.map(season_map)

    def _dist(group_col: str, order: list[str]) -> pd.DataFrame:
        t = pd.crosstab(df[group_col], df["label"], normalize="index" if normalize else False)
        t = t.reindex(order)
        return t.fillna(0.0)

    week_dist = _dist("weekpart", ["weekday", "weekend"])
    season_dist = _dist("season", ["Winter", "Spring", "Summer", "Fall"])
    month_dist = _dist("month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    labels_sorted = sorted(df["label"].dropna().unique().tolist())
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=("Weekday vs Weekend", "Season", "Month"),
    )

    for lab in labels_sorted:
        fig.add_trace(
            go.Bar(x=week_dist.index, y=week_dist.get(lab, 0.0), name=lab, legendgroup=lab),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=season_dist.index, y=season_dist.get(lab, 0.0), name=lab, legendgroup=lab, showlegend=False),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=month_dist.index, y=month_dist.get(lab, 0.0), name=lab, legendgroup=lab, showlegend=False),
            row=3,
            col=1,
        )

    y_title = "Share" if normalize else "Count"
    fig.update_layout(
        title=title,
        barmode="stack",
        template="plotly_white",
        height=height,
        legend_title_text="DTW label",
    )
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_yaxes(title_text=y_title, row=2, col=1)
    fig.update_yaxes(title_text=y_title, row=3, col=1)
    fig.update_xaxes(title_text="Weekpart", row=1, col=1)
    fig.update_xaxes(title_text="Season", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.show()
    return fig


def animate_city_consumption_map(
    city,
    gdf_city,
    fsa_col: str = "FSA",
    freq: str = "D",
    agg: str = "mean",
    start=None,
    end=None,
    title_prefix: str = "City Electricity Animation",
    weather_df: pd.DataFrame | None = None,
    weather_dt_col: str = "date_time_local",
    weather_temp_col: str = "temperature",
    weather_agg: str = "mean",
    weather_alpha: float = 0.18,
    weather_colorscale: str = "RdBu_r",
    show_weather_colorbar: bool = False,
):
    """
    Animate city FSA consumption from City object electricity series.

    Parameters
    ----------
    city : src.domain.city.City
        City object with FSA electricity series.
    gdf_city : geopandas.GeoDataFrame
        City FSA geometry table.
    fsa_col : str
        Geometry column containing FSA code.
    freq : str
        Resample frequency (e.g., "H", "D", "W").
    agg : str
        Aggregation method: "mean" or "sum".
    start, end : str or pd.Timestamp or None
        Optional date range filter applied before resampling.
    weather_df : pd.DataFrame | None
        Optional weather table. When provided, a low-alpha city background layer
        is colored by temperature and animated across time.
    weather_dt_col : str
        Weather datetime column name in weather_df.
    weather_temp_col : str
        Weather temperature column name in weather_df.
    weather_agg : str
        Weather aggregation method after resampling: "mean", "min", or "max".
    weather_alpha : float
        Opacity for weather background layer in [0, 1].
    weather_colorscale : str
        Plotly colorscale name for weather background.
    show_weather_colorbar : bool
        If True, show separate weather colorbar.
    """
    elec_df = city.electricity_frame()
    if elec_df.empty:
        raise ValueError(f"City '{city.name}' has no FSA electricity series.")

    gdf = gdf_city.copy().to_crs(epsg=4326)
    if fsa_col not in gdf.columns:
        raise KeyError(f"'{fsa_col}' not found in geometry columns: {list(gdf.columns)}")

    elec_df.index = pd.to_datetime(elec_df.index, errors="coerce")
    elec_df = elec_df.dropna(how="all").sort_index()

    if start is not None or end is not None:
        elec_df = elec_df.loc[start:end]
    if elec_df.empty:
        raise ValueError("No electricity data available after applying date range.")

    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be one of: 'mean', 'sum'.")
    if weather_agg not in {"mean", "min", "max"}:
        raise ValueError("weather_agg must be one of: 'mean', 'min', 'max'.")
    if not (0.0 <= float(weather_alpha) <= 1.0):
        raise ValueError("weather_alpha must be in [0, 1].")
    elec_agg = elec_df.resample(freq).sum() if agg == "sum" else elec_df.resample(freq).mean()
    elec_agg = elec_agg.dropna(how="all")
    if elec_agg.empty:
        raise ValueError("No data available after resampling.")

    fsas_geo = set(gdf[fsa_col].astype(str))
    fsas = [c for c in elec_agg.columns if str(c) in fsas_geo]
    if not fsas:
        raise ValueError("No overlapping FSA codes between city object and geometry.")
    elec_agg = elec_agg[fsas]
    dates = elec_agg.index

    geojson = json.loads(gdf.to_json())
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    centroid = geom.centroid
    center = {"lat": float(centroid.y), "lon": float(centroid.x)}

    weather_series = None
    bg_geojson = None
    weather_zmin = None
    weather_zmax = None
    bg_location = "__city_background__"
    if weather_df is not None:
        if weather_dt_col not in weather_df.columns:
            raise KeyError(
                f"'{weather_dt_col}' not found in weather_df columns: {list(weather_df.columns)}"
            )
        if weather_temp_col not in weather_df.columns:
            raise KeyError(
                f"'{weather_temp_col}' not found in weather_df columns: {list(weather_df.columns)}"
            )

        w = weather_df[[weather_dt_col, weather_temp_col]].copy()
        w[weather_dt_col] = pd.to_datetime(w[weather_dt_col], errors="coerce")
        w[weather_temp_col] = pd.to_numeric(w[weather_temp_col], errors="coerce")
        w = w.dropna(subset=[weather_dt_col, weather_temp_col]).sort_values(weather_dt_col)
        if w.empty:
            raise ValueError("weather_df has no valid datetime/temperature rows after cleaning.")

        w = w.set_index(weather_dt_col)[weather_temp_col]
        if start is not None or end is not None:
            w = w.loc[start:end]
        if w.empty:
            raise ValueError("No weather data available after applying date range.")

        if weather_agg == "mean":
            w_agg = w.resample(freq).mean()
        elif weather_agg == "min":
            w_agg = w.resample(freq).min()
        else:
            w_agg = w.resample(freq).max()
        w_agg = w_agg.dropna()
        if w_agg.empty:
            raise ValueError("No weather data available after resampling.")

        # Align weather values to electricity animation timestamps (timezone-safe).
        dt_anim = pd.DatetimeIndex(dates)
        dt_anim_align = dt_anim.tz_convert("UTC").tz_localize(None) if dt_anim.tz is not None else dt_anim
        w_idx = pd.DatetimeIndex(w_agg.index)
        w_idx_align = w_idx.tz_convert("UTC").tz_localize(None) if w_idx.tz is not None else w_idx
        w_aligned = pd.Series(w_agg.values, index=w_idx_align).sort_index()
        weather_series = w_aligned.reindex(dt_anim_align, method="nearest")
        weather_series.index = dt_anim

        weather_vals = weather_series.to_numpy(dtype=float)
        weather_zmin = float(np.nanpercentile(weather_vals, 2))
        weather_zmax = float(np.nanpercentile(weather_vals, 98))
        if not np.isfinite(weather_zmin) or not np.isfinite(weather_zmax) or weather_zmin == weather_zmax:
            weather_zmin = float(np.nanmin(weather_vals))
            weather_zmax = float(np.nanmax(weather_vals))
            if weather_zmin == weather_zmax:
                weather_zmin = weather_zmin - 1.0
                weather_zmax = weather_zmax + 1.0

        bg_gdf = gdf[["geometry"]].copy()
        bg_geom = bg_gdf.geometry.union_all() if hasattr(bg_gdf.geometry, "union_all") else bg_gdf.geometry.unary_union
        bg_gdf = bg_gdf.iloc[:1].copy()
        bg_gdf["geometry"] = [bg_geom]
        bg_gdf["city_bg"] = bg_location
        bg_geojson = json.loads(bg_gdf.to_json())

    zvals = elec_agg.to_numpy(dtype=float)
    zmin = float(np.nanpercentile(zvals, 2))
    zmax = float(np.nanpercentile(zvals, 98))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin == zmax:
        zmin = float(np.nanmin(zvals))
        zmax = float(np.nanmax(zvals))

    def _trace_at(ts):
        s = elec_agg.loc[ts]
        locs = [str(x) for x in s.index]
        return go.Choroplethmap(
            geojson=geojson,
            locations=locs,
            z=s.values.astype(float),
            featureidkey=f"properties.{fsa_col}",
            colorscale="Turbo",
            zmin=zmin,
            zmax=zmax,
            marker_line_width=0.4,
            colorbar={"title": "kWh"},
            customdata=np.array(locs),
            hovertemplate="FSA: %{customdata}<br>kWh: %{z:.2f}<extra></extra>",
        )

    def _weather_trace_at(ts):
        temp = float(weather_series.loc[ts]) if weather_series is not None else np.nan
        return go.Choroplethmap(
            geojson=bg_geojson,
            locations=[bg_location],
            z=[temp],
            featureidkey="properties.city_bg",
            colorscale=weather_colorscale,
            zmin=weather_zmin,
            zmax=weather_zmax,
            marker_opacity=float(weather_alpha),
            marker_line_width=0.0,
            showscale=bool(show_weather_colorbar),
            colorbar={"title": "Temp (C)"},
            customdata=np.array([city.name]),
            hovertemplate="City: %{customdata}<br>Temp: %{z:.2f} C<extra></extra>",
        )

    if weather_series is None:
        fig = go.Figure(data=[_trace_at(dates[0])])
        fig.frames = [go.Frame(name=str(ts), data=[_trace_at(ts)]) for ts in dates]
    else:
        fig = go.Figure(data=[_weather_trace_at(dates[0]), _trace_at(dates[0])])
        fig.frames = [go.Frame(name=str(ts), data=[_weather_trace_at(ts), _trace_at(ts)]) for ts in dates]

    slider_steps = [
        {
            "args": [[str(ts)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M" if freq.upper().startswith("H") else "%Y-%m-%d"),
            "method": "animate",
        }
        for ts in dates
    ]

    fig.update_layout(
        title=f"{title_prefix}: {city.name}",
        map=dict(style="carto-positron", center=center, zoom=8),
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 0.01,
                "xanchor": "left",
                "yanchor": "bottom",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play 1x",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 600, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Play 2x",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Play 3x",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.01,
                "len": 0.9,
                "xanchor": "left",
                "yanchor": "bottom",
                "steps": slider_steps,
                "currentvalue": {"prefix": "Time: "},
            }
        ],
    )
    return fig


def plot_imputation_holdout_monthly_comparison(
    elec_true: pd.DataFrame,
    elec_imputed: pd.DataFrame,
    fsas=None,
    *,
    freq: str = "MS",
    agg: str = "sum",
    max_fsas: int = 6,
    height_per_fsa: int = 260,
    title: str = "Imputation Holdout: Monthly True vs Imputed",
):
    """
    Plot monthly true vs imputed series for holdout FSA validation.

    Parameters
    ----------
    elec_true : pd.DataFrame
        Ground-truth wide dataframe (index=DatetimeIndex, columns=FSA).
    elec_imputed : pd.DataFrame
        Imputed wide dataframe aligned to same style as elec_true.
    fsas : str | list[str] | None
        FSA code(s) to plot. If None, uses overlap columns up to max_fsas.
    freq : str
        Resample frequency for comparison (default "MS").
    agg : str
        Aggregation method: "sum" or "mean".
    max_fsas : int
        Maximum FSAs when fsas is None.
    height_per_fsa : int
        Height per subplot row in pixels.
    title : str
        Figure title.
    """
    if not isinstance(elec_true.index, pd.DatetimeIndex):
        raise TypeError("elec_true index must be a DatetimeIndex.")
    if not isinstance(elec_imputed.index, pd.DatetimeIndex):
        raise TypeError("elec_imputed index must be a DatetimeIndex.")
    if agg not in {"sum", "mean"}:
        raise ValueError("agg must be one of: 'sum', 'mean'.")

    overlap = [c for c in elec_true.columns if c in elec_imputed.columns]
    if not overlap:
        raise ValueError("No overlapping FSA columns between elec_true and elec_imputed.")

    if fsas is None:
        fsas = overlap[:max_fsas]
    elif isinstance(fsas, str):
        fsas = [fsas]
    else:
        fsas = [str(c) for c in fsas]

    missing = [c for c in fsas if c not in overlap]
    if missing:
        raise KeyError(f"Requested FSAs not found in overlap columns: {missing}")
    if not fsas:
        raise ValueError("No FSAs selected for plotting.")

    t = elec_true[fsas].sort_index()
    p = elec_imputed[fsas].sort_index()
    t_m = t.resample(freq).sum() if agg == "sum" else t.resample(freq).mean()
    p_m = p.resample(freq).sum() if agg == "sum" else p.resample(freq).mean()

    fig = make_subplots(
        rows=len(fsas),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"FSA {fsa}" for fsa in fsas],
    )

    for i, fsa in enumerate(fsas, start=1):
        pair = pd.concat(
            [t_m[fsa].rename("true"), p_m[fsa].rename("imputed")],
            axis=1,
        ).dropna(how="all")
        if pair.empty:
            continue

        denom = pair["true"].replace(0, np.nan).abs()
        ape = (pair["imputed"] - pair["true"]).abs() / denom * 100.0

        fig.add_trace(
            go.Scatter(
                x=pair.index,
                y=pair["true"],
                mode="lines+markers",
                name="True",
                legendgroup="true",
                showlegend=(i == 1),
                line=dict(color="#1f77b4", width=2),
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pair.index,
                y=pair["imputed"],
                mode="lines+markers",
                name="Imputed",
                legendgroup="imputed",
                showlegend=(i == 1),
                line=dict(color="#d62728", width=2, dash="dot"),
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pair.index,
                y=pair["imputed"],
                mode="markers",
                name="APE %",
                legendgroup="ape",
                showlegend=(i == 1),
                marker=dict(color="#2ca02c", size=7, symbol="diamond"),
                customdata=np.c_[ape.fillna(np.nan).to_numpy()],
                hovertemplate=(
                    "date=%{x|%Y-%m}<br>"
                    "imputed=%{y:.2f}<br>"
                    "APE=%{customdata[0]:.2f}%<extra></extra>"
                ),
            ),
            row=i,
            col=1,
        )

        fig.update_yaxes(title_text="kWh", row=i, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(350, int(height_per_fsa * len(fsas))),
    )
    fig.update_xaxes(title_text="Month", row=len(fsas), col=1)
    fig.show()
    return fig
