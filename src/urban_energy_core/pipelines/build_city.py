from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.urban_energy_core.domain.city import City
from src.urban_energy_core.domain.fsa import FSA


def _pick_fsa_column(gdf, candidates: list[str] | None = None) -> str:
    candidates = candidates or ["FSA", "CFSAUID", "CFSAUID21", "GEO_UID", "GEO UID", "CP3"]
    for c in candidates:
        if c in gdf.columns:
            return c
    raise KeyError(f"No FSA code column found in geometry columns: {list(gdf.columns)}")


def _city_boundary_from_gdf(gdf):
    try:
        return gdf.unary_union
    except Exception:
        return None


def _geometry_union(g_part):
    if g_part is None or len(g_part) == 0:
        return None
    if hasattr(g_part, "unary_union"):
        try:
            return g_part.unary_union
        except Exception:
            return None
    return None


def _census_row_for_fsa(
    census_df: pd.DataFrame | None,
    fsa_code: str,
    census_key_col: str | None = None,
) -> pd.Series | None:
    if census_df is None:
        return None
    if census_key_col is not None and census_key_col in census_df.columns:
        match = census_df[census_df[census_key_col].astype(str) == str(fsa_code)]
        if len(match) > 0:
            return match.iloc[0]
        return None
    if fsa_code in census_df.index:
        row = census_df.loc[fsa_code]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row
    return None


def build_cities_from_data(
    elec_df: pd.DataFrame,
    city_geojsons: dict[str, Any],
    city_weather: dict[str, pd.DataFrame] | None = None,
    census_df: pd.DataFrame | None = None,
    city_to_fsa_codes: dict[str, list[str]] | None = None,
    geometry_fsa_col: str | None = None,
    census_key_col: str | None = None,
    show_progress: bool = True,
) -> dict[str, City]:
    """
    Build City objects composed of FSA objects from loaded dataframes.
    """
    cities: dict[str, City] = {}
    city_items = list(city_geojsons.items())
    city_iter = tqdm(city_items, desc="Building city objects") if show_progress else city_items

    for city_name, gdf in city_iter:
        weather_df = city_weather.get(city_name) if city_weather is not None else None
        city = City(
            name=city_name,
            boundary=_city_boundary_from_gdf(gdf),
            weather=weather_df,
        )

        if city_to_fsa_codes and city_name in city_to_fsa_codes:
            fsa_codes = [str(x) for x in city_to_fsa_codes[city_name]]
            gdf_by_fsa = {}
            fsa_col = _pick_fsa_column(gdf) if geometry_fsa_col is None else geometry_fsa_col
            for code in fsa_codes:
                g_part = gdf[gdf[fsa_col].astype(str) == code]
                if len(g_part) > 0:
                    gdf_by_fsa[code] = g_part
        else:
            fsa_col = _pick_fsa_column(gdf) if geometry_fsa_col is None else geometry_fsa_col
            fsa_codes = sorted(gdf[fsa_col].astype(str).unique().tolist())
            gdf_by_fsa = {code: gdf[gdf[fsa_col].astype(str) == code] for code in fsa_codes}

        fsa_codes = [code for code in fsa_codes if code in elec_df.columns]
        for code in fsa_codes:
            g_part = gdf_by_fsa.get(code)
            geometry = _geometry_union(g_part)
            census_row = _census_row_for_fsa(census_df, code, census_key_col=census_key_col)
            fsa = FSA(
                code=code,
                geometry=geometry,
                electricity=elec_df[code].copy(),
                census=census_row,
            )
            city.add_fsa(fsa)
        cities[city_name] = city

    return cities
