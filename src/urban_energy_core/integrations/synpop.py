from __future__ import annotations

from typing import Any

import pandas as pd

from urban_energy_core.domain.city import City


def _safe_population_value(census: pd.Series | dict | None, population_col: str) -> float | None:
    if census is None:
        return None
    try:
        value = census[population_col]
    except Exception:
        return None
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def build_synpop_da_input_table(
    city: City,
    *,
    population_col: str = "Population and dwelling counts / Population, 2021",
    include_geometry: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for da in city.das.values():
        row: dict[str, Any] = {
            "city_name": city.name,
            "da_code": da.code,
            "population_2021": _safe_population_value(da.census, population_col),
            "nearest_fsa_1": da.nearest_fsas[0] if len(da.nearest_fsas) > 0 else None,
            "nearest_fsa_2": da.nearest_fsas[1] if len(da.nearest_fsas) > 1 else None,
            "nearest_fsa_3": da.nearest_fsas[2] if len(da.nearest_fsas) > 2 else None,
            "has_census": da.census is not None,
            "has_geometry": da.geometry is not None,
            "city_crs": city.crs,
        }
        if include_geometry:
            row["geometry"] = da.geometry
        rows.append(row)
    return pd.DataFrame(rows).sort_values("da_code").reset_index(drop=True) if rows else pd.DataFrame()


def build_synpop_city_manifest(
    city: City,
    *,
    population_col: str = "Population and dwelling counts / Population, 2021",
) -> dict[str, Any]:
    da_table = build_synpop_da_input_table(city, population_col=population_col, include_geometry=False)
    total_pop = pd.to_numeric(da_table.get("population_2021"), errors="coerce").sum(min_count=1)
    if pd.isna(total_pop):
        total_pop = None
    else:
        total_pop = float(total_pop)
    return {
        "city_name": city.name,
        "city_crs": city.crs,
        "n_fsas": len(city.fsas),
        "n_das": len(city.das),
        "n_buildings": len(city.buildings),
        "n_das_with_census": int(da_table.get("has_census", pd.Series(dtype=bool)).sum()) if not da_table.empty else 0,
        "n_das_with_geometry": int(da_table.get("has_geometry", pd.Series(dtype=bool)).sum()) if not da_table.empty else 0,
        "population_col": population_col,
        "population_2021_sum": total_pop,
        "da_codes": da_table["da_code"].tolist() if not da_table.empty else [],
    }


def summarize_synpop_outputs_by_da(
    synpop_df: pd.DataFrame,
    *,
    da_col: str = "area",
    household_id_col: str = "HID",
) -> pd.DataFrame:
    if da_col not in synpop_df.columns:
        raise KeyError(f"DA column '{da_col}' not found in synthetic population output.")

    work = synpop_df.copy()
    work["da_code"] = work[da_col].astype(str).str.strip()

    grouped = work.groupby("da_code", sort=True)
    summary = pd.DataFrame({"da_code": sorted(grouped.groups.keys())})
    summary["n_individuals_syn"] = summary["da_code"].map(grouped.size())

    if household_id_col in work.columns:
        hh = work.loc[work[household_id_col].notna()].copy()
        hh = hh.loc[hh[household_id_col] != -1]
        hh["da_code"] = hh["da_code"].astype(str)
        hh_counts = hh.groupby("da_code")[household_id_col].nunique()
        summary["n_households_syn"] = summary["da_code"].map(hh_counts).fillna(0).astype(int)

    return summary


def merge_synpop_summary_to_da_input(
    da_input_table: pd.DataFrame,
    synpop_summary_df: pd.DataFrame,
    *,
    da_col: str = "da_code",
) -> pd.DataFrame:
    if da_col not in da_input_table.columns:
        raise KeyError(f"DA input table is missing '{da_col}'.")
    if da_col not in synpop_summary_df.columns:
        raise KeyError(f"Synthetic population summary is missing '{da_col}'.")
    return da_input_table.merge(synpop_summary_df, on=da_col, how="left")
