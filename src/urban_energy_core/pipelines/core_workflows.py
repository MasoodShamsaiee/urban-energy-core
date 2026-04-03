from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urban_energy_core.io.load_data import (
    load_all_da_census,
    load_all_fsa_census,
    load_and_prepare_electricity_4cities,
    load_city_da_geojsons,
    load_city_fsa_geojsons,
    load_city_weather_csvs,
    load_processed_da_electricity_wide,
    load_processed_electricity_wide,
    save_processed_electricity_wide,
)
from urban_energy_core.pipelines.build_city import build_cities_from_data
from urban_energy_core.services.anomalies import treat_anomalies_until_target_rate
from urban_energy_core.services.imputation import impute_missing_fsa_energy_by_census_proximity
from urban_energy_core.services.preprocess import preprocess_wide_fsa_timeseries


@dataclass(frozen=True)
class CoreProjectData:
    census_df: pd.DataFrame
    geo: dict[str, Any]
    weather: dict[str, pd.DataFrame]
    da_census_df: pd.DataFrame | None = None
    da_geo: dict[str, Any] | None = None


@dataclass(frozen=True)
class CityBuildResult:
    elec_df: pd.DataFrame
    census_df: pd.DataFrame
    geo: dict[str, Any]
    weather: dict[str, pd.DataFrame]
    cities: dict[str, Any]
    da_elec_df: pd.DataFrame | None = None
    da_census_df: pd.DataFrame | None = None
    da_geo: dict[str, Any] | None = None


@dataclass(frozen=True)
class ElectricityRebuildResult:
    elec_raw: pd.DataFrame
    elec_preprocessed: pd.DataFrame
    elec_clean: pd.DataFrame
    elec_weather_normalized: pd.DataFrame
    elec_per_capita: pd.DataFrame
    elec_per_capita_imputed: pd.DataFrame
    elec_rebuilt: pd.DataFrame
    cities: dict[str, Any]
    imputation_report: pd.DataFrame
    feature_scores: pd.DataFrame
    qc_pre: pd.DataFrame
    conformance_report: pd.DataFrame
    output_path: Path | None


def project_root(cwd: Path | None = None) -> Path:
    here = (cwd or Path.cwd()).resolve()
    return here.parent if here.name.lower() == "notebooks" else here


def clean_weather_tables(weather: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for city, df in weather.items():
        work = df.copy()
        work["date_time_local"] = pd.to_datetime(work["date_time_local"], errors="coerce")
        work["temperature"] = pd.to_numeric(work["temperature"], errors="coerce")
        work = work.dropna(subset=["date_time_local", "temperature"]).sort_values("date_time_local")
        out[city] = work
    return out


def load_core_project_data(
    *,
    census_drop_key_col: bool = False,
    census_index_col: str = "GEO UID",
    load_da: bool = False,
    da_census_drop_key_col: bool = False,
    da_census_index_col: str = "DAUID",
    show_progress: bool = True,
) -> CoreProjectData:
    census_df = load_all_fsa_census(drop_key_col=census_drop_key_col, show_progress=show_progress)
    if census_index_col in census_df.columns:
        census_df = census_df.set_index(census_index_col)
    census_df.index = census_df.index.astype(str)

    geo = load_city_fsa_geojsons(show_progress=show_progress)
    weather = clean_weather_tables(load_city_weather_csvs(show_progress=show_progress))
    da_census_df = None
    da_geo = None
    if load_da:
        da_census_df = load_all_da_census(
            drop_key_col=da_census_drop_key_col,
            show_progress=show_progress,
        )
        if da_census_index_col in da_census_df.columns:
            da_census_df = da_census_df.set_index(da_census_index_col)
        da_census_df.index = da_census_df.index.astype(str)
        da_geo = load_city_da_geojsons(show_progress=show_progress)

    return CoreProjectData(
        census_df=census_df,
        geo=geo,
        weather=weather,
        da_census_df=da_census_df,
        da_geo=da_geo,
    )


def build_city_bundle_from_processed_electricity(
    elec_path: str | Path,
    *,
    da_elec_path: str | Path | None = None,
    load_da: bool = False,
    show_progress: bool = True,
) -> CityBuildResult:
    core = load_core_project_data(load_da=load_da, show_progress=show_progress)
    elec_df = load_processed_electricity_wide(elec_path).sort_index()
    da_elec_df = None
    if load_da or da_elec_path is not None:
        try:
            da_elec_df = load_processed_da_electricity_wide(da_elec_path).sort_index()
        except FileNotFoundError:
            da_elec_df = None
    cities = build_cities_from_data(
        elec_df=elec_df,
        city_geojsons=core.geo,
        city_weather=core.weather,
        census_df=core.census_df,
        da_elec_df=da_elec_df,
        city_da_geojsons=core.da_geo,
        da_census_df=core.da_census_df,
        show_progress=show_progress,
    )
    return CityBuildResult(
        elec_df=elec_df,
        census_df=core.census_df,
        geo=core.geo,
        weather=core.weather,
        cities=cities,
        da_elec_df=da_elec_df,
        da_census_df=core.da_census_df,
        da_geo=core.da_geo,
    )


def _geometry_fsa_codes(city_geojsons: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for gdf in city_geojsons.values():
        for candidate in ["FSA", "CFSAUID", "CFSAUID21", "GEO_UID", "GEO UID", "CP3"]:
            if candidate in gdf.columns:
                codes.update(gdf[candidate].astype(str).dropna().tolist())
                break
    return sorted(codes)


def rebuild_electricity_with_weather_and_imputation(
    *,
    raw_energy_path: str | Path | None = None,
    output_path: str | Path | None = None,
    population_col: str = "Population and dwelling counts / Population, 2021",
    preprocess_kwargs: dict[str, Any] | None = None,
    anomaly_kwargs: dict[str, Any] | None = None,
    imputation_kwargs: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> ElectricityRebuildResult:
    preprocess_kwargs = {
        "tz_local": "America/Toronto",
        "freq": None,
        "min_coverage": 0.90,
        "clip_negatives": True,
        "fill_method": None,
        **(preprocess_kwargs or {}),
    }
    anomaly_kwargs = {
        "target_rate": 0.001,
        "period": 24,
        "z_thresh": 3.0,
        "robust": False,
        "max_iter": 10,
        "method": "interp",
        "show_progress": show_progress,
        **(anomaly_kwargs or {}),
    }
    imputation_kwargs = {"show_progress": show_progress, **(imputation_kwargs or {})}

    core = load_core_project_data(show_progress=show_progress)
    elec_raw = load_and_prepare_electricity_4cities(path=raw_energy_path)
    elec_preprocessed, qc_pre = preprocess_wide_fsa_timeseries(elec_raw, **preprocess_kwargs)
    elec_clean, _, conformance_report = treat_anomalies_until_target_rate(
        elec_df=elec_preprocessed,
        **anomaly_kwargs,
    )

    cities = build_cities_from_data(
        elec_df=elec_clean,
        city_geojsons=core.geo,
        city_weather=core.weather,
        census_df=core.census_df,
        show_progress=show_progress,
    )

    normalized_by_city = {}
    for city_name, city in cities.items():
        normalized_by_city[city_name] = city.normalize_all_fsas_for_weather(
            dt_col="date_time_local",
            temp_col="temperature",
            in_place=False,
            show_progress=show_progress,
        )
    elec_norm = pd.concat(normalized_by_city.values(), axis=1)
    elec_norm = elec_norm.loc[:, ~elec_norm.columns.duplicated()].sort_index()

    pop_series = pd.to_numeric(core.census_df.get(population_col), errors="coerce")
    pop_series.index = pop_series.index.astype(str)

    elec_pc = elec_norm.copy()
    for col in elec_pc.columns:
        pop = pop_series.get(str(col), np.nan)
        elec_pc[col] = elec_pc[col] / float(pop) if pd.notna(pop) and float(pop) > 0 else np.nan

    elec_pc_imputed, imputation_report, feature_scores = impute_missing_fsa_energy_by_census_proximity(
        elec_df=elec_pc,
        census_df=core.census_df,
        geometry_fsas=_geometry_fsa_codes(core.geo),
        population_col=population_col,
        **imputation_kwargs,
    )

    elec_rebuilt = elec_pc_imputed.copy()
    for col in elec_rebuilt.columns:
        pop = pop_series.get(str(col), np.nan)
        elec_rebuilt[col] = elec_rebuilt[col] * float(pop) if pd.notna(pop) and float(pop) > 0 else np.nan

    saved_path = None
    if output_path is not None:
        saved_path = save_processed_electricity_wide(elec_rebuilt, output_path)

    return ElectricityRebuildResult(
        elec_raw=elec_raw,
        elec_preprocessed=elec_preprocessed,
        elec_clean=elec_clean,
        elec_weather_normalized=elec_norm,
        elec_per_capita=elec_pc,
        elec_per_capita_imputed=elec_pc_imputed,
        elec_rebuilt=elec_rebuilt,
        cities=cities,
        imputation_report=imputation_report,
        feature_scores=feature_scores,
        qc_pre=qc_pre,
        conformance_report=conformance_report,
        output_path=saved_path,
    )


def compute_and_attach_city_tables(
    city,
    *,
    unit: str = "fsa",
    prism_kwargs: dict[str, Any] | None = None,
    short_term_kwargs: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    prism_kwargs = {
        "unit": unit,
        "per_capita": True,
        "weather_normalized": False,
        "mode": "segmented",
        "show_progress": True,
        **(prism_kwargs or {}),
    }
    short_term_kwargs = {
        "unit": unit,
        "per_capita": True,
        "weather_normalized": False,
        "winter_only": True,
        "weekday_only": False,
        "aggregate": True,
        "show_progress": True,
        **(short_term_kwargs or {}),
    }
    prism_df = city.compute_prism_table(**prism_kwargs)
    short_term_df = city.compute_short_term_table(**short_term_kwargs)

    table_prefix = "da" if unit == "da" else "fsa"
    setattr(city, f"{table_prefix}_prism_table", prism_df.copy())
    setattr(city, f"{table_prefix}_short_term_table", short_term_df.copy())
    if unit == "fsa":
        city.prism_table = prism_df.copy()
        city.short_term_table = short_term_df.copy()

    getter = city.get_da if unit == "da" else city.get_fsa
    for fsa_code, row in prism_df.iterrows():
        getter(fsa_code).prism_summary = row.to_dict()
    for fsa_code, row in short_term_df.iterrows():
        getter(fsa_code).short_term_summary = row.to_dict()

    return {"prism_table": prism_df, "short_term_table": short_term_df}
