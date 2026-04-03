from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from urban_energy_core.domain.city import City
from urban_energy_core.domain.building import Building
from urban_energy_core.domain.da import DA
from urban_energy_core.domain.fsa import FSA


def _pick_fsa_column(gdf, candidates: list[str] | None = None) -> str:
    candidates = candidates or ["FSA", "CFSAUID", "CFSAUID21", "GEO_UID", "GEO UID", "CP3"]
    for c in candidates:
        if c in gdf.columns:
            return c
    raise KeyError(f"No FSA code column found in geometry columns: {list(gdf.columns)}")


def _pick_da_column(gdf, candidates: list[str] | None = None) -> str:
    candidates = candidates or ["DAUID", "DAUID21", "DA", "DisseminationArea", "GEO_UID", "GEO UID"]
    for c in candidates:
        if c in gdf.columns:
            return c
    raise KeyError(f"No DA code column found in geometry columns: {list(gdf.columns)}")


def _city_boundary_from_gdf(gdf):
    try:
        return gdf.unary_union
    except Exception:
        return None


def _gdf_crs_string(gdf) -> str | None:
    crs = getattr(gdf, "crs", None)
    if crs is None:
        return None
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def _geometry_union(g_part):
    if g_part is None or len(g_part) == 0:
        return None
    if hasattr(g_part, "unary_union"):
        try:
            return g_part.unary_union
        except Exception:
            return None
    if hasattr(g_part, "columns") and "geometry" in g_part.columns:
        geometries = [geom for geom in g_part["geometry"].tolist() if geom is not None]
        if not geometries:
            return None
        if len(geometries) == 1:
            return geometries[0]
        try:
            from shapely.ops import unary_union

            return unary_union(geometries)
        except Exception:
            return geometries[0]
    return None


def _census_row_for_code(
    census_df: pd.DataFrame | None,
    code: str,
    census_key_col: str | None = None,
) -> pd.Series | None:
    if census_df is None:
        return None
    if census_key_col is not None and census_key_col in census_df.columns:
        match = census_df[census_df[census_key_col].astype(str) == str(code)]
        if len(match) > 0:
            return match.iloc[0]
        return None
    if code in census_df.index:
        row = census_df.loc[code]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row
    return None


def _group_geometry_by_code(gdf, code_col: str, codes: list[str] | None = None) -> tuple[list[str], dict[str, Any]]:
    if codes is None:
        codes = sorted(gdf[code_col].astype(str).unique().tolist())
    grouped = {}
    for code in codes:
        g_part = gdf[gdf[code_col].astype(str) == str(code)]
        if len(g_part) > 0:
            grouped[str(code)] = g_part
    return [str(code) for code in codes], grouped


def _pick_building_column(df, candidates: list[str] | None = None) -> str:
    candidates = candidates or ["building_id", "ID_UEV", "OBJECTID", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No building code column found in columns: {list(df.columns)}")


def build_cities_from_data(
    elec_df: pd.DataFrame,
    city_geojsons: dict[str, Any],
    city_weather: dict[str, pd.DataFrame] | None = None,
    census_df: pd.DataFrame | None = None,
    city_to_fsa_codes: dict[str, list[str]] | None = None,
    geometry_fsa_col: str | None = None,
    census_key_col: str | None = None,
    da_elec_df: pd.DataFrame | None = None,
    city_da_geojsons: dict[str, Any] | None = None,
    da_census_df: pd.DataFrame | None = None,
    city_to_da_codes: dict[str, list[str]] | None = None,
    geometry_da_col: str | None = None,
    da_census_key_col: str | None = None,
    da_nearest_fsa_k: int | None = 3,
    city_building_gdfs: dict[str, Any] | None = None,
    building_code_col: str | None = None,
    assign_building_units: bool = True,
    show_progress: bool = True,
) -> dict[str, City]:
    """
    Build City objects composed of FSA objects and optional DA objects from loaded dataframes.
    """
    cities: dict[str, City] = {}
    city_items = list(city_geojsons.items())
    city_iter = tqdm(city_items, desc="Building city objects") if show_progress else city_items

    for city_name, gdf in city_iter:
        weather_df = city_weather.get(city_name) if city_weather is not None else None
        city = City(
            name=city_name,
            boundary=_city_boundary_from_gdf(gdf),
            crs=_gdf_crs_string(gdf),
            weather=weather_df,
        )

        if city_to_fsa_codes and city_name in city_to_fsa_codes:
            fsa_col = _pick_fsa_column(gdf) if geometry_fsa_col is None else geometry_fsa_col
            requested_codes = [str(x) for x in city_to_fsa_codes[city_name]]
            fsa_codes, gdf_by_fsa = _group_geometry_by_code(gdf, fsa_col, requested_codes)
        else:
            fsa_col = _pick_fsa_column(gdf) if geometry_fsa_col is None else geometry_fsa_col
            fsa_codes, gdf_by_fsa = _group_geometry_by_code(gdf, fsa_col)

        fsa_codes = [code for code in fsa_codes if code in elec_df.columns]
        for code in fsa_codes:
            g_part = gdf_by_fsa.get(code)
            geometry = _geometry_union(g_part)
            census_row = _census_row_for_code(census_df, code, census_key_col=census_key_col)
            fsa = FSA(
                code=code,
                geometry=geometry,
                electricity=elec_df[code].copy(),
                census=census_row,
            )
            city.add_fsa(fsa)

        if city_da_geojsons is not None and city_name in city_da_geojsons:
            da_gdf = city_da_geojsons[city_name]
            if city_to_da_codes and city_name in city_to_da_codes:
                da_col = _pick_da_column(da_gdf) if geometry_da_col is None else geometry_da_col
                requested_da_codes = [str(x) for x in city_to_da_codes[city_name]]
                da_codes, gdf_by_da = _group_geometry_by_code(da_gdf, da_col, requested_da_codes)
            else:
                da_col = _pick_da_column(da_gdf) if geometry_da_col is None else geometry_da_col
                da_codes, gdf_by_da = _group_geometry_by_code(da_gdf, da_col)

            if da_elec_df is not None:
                da_codes = [code for code in da_codes if code in da_elec_df.columns]
            for code in da_codes:
                g_part = gdf_by_da.get(code)
                geometry = _geometry_union(g_part)
                census_row = _census_row_for_code(
                    da_census_df,
                    code,
                    census_key_col=da_census_key_col,
                )
                da = DA(
                    code=code,
                    geometry=geometry,
                    electricity=da_elec_df[code].copy() if da_elec_df is not None and code in da_elec_df.columns else None,
                    census=census_row,
                )
                city.add_da(da)

            if da_nearest_fsa_k is not None and city.das:
                city.rank_da_to_fsa_distances(max_neighbors=da_nearest_fsa_k)

        if city_building_gdfs is not None and city_name in city_building_gdfs:
            building_df = city_building_gdfs[city_name]
            code_col = _pick_building_column(building_df) if building_code_col is None else building_code_col
            for _, row in building_df.iterrows():
                code = str(row[code_col]).strip()
                if not code:
                    continue

                aliases = row.get("aliases") if isinstance(row.get("aliases"), dict) else {}
                for key, value in row.items():
                    if str(key).startswith("alias_") and not pd.isna(value):
                        aliases[str(key)[6:]] = value

                provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
                for key, value in row.items():
                    if str(key).startswith("provenance_") and not pd.isna(value):
                        provenance[str(key)[11:]] = value

                metadata = {
                    k: v
                    for k, v in row.items()
                    if k not in {
                        code_col,
                        "geometry",
                        "building_id",
                        "building_type",
                        "building_category",
                        "year_built",
                        "num_dwellings",
                        "stories",
                        "footprint_area",
                        "total_floor_area",
                        "volume",
                        "height_m",
                        "aliases",
                        "provenance",
                        "fsa_code",
                        "da_code",
                    }
                    and not str(k).startswith("alias_")
                    and not str(k).startswith("provenance_")
                }
                building = Building(
                    code=code,
                    geometry=row.get("geometry"),
                    fsa_code=row.get("fsa_code"),
                    da_code=row.get("da_code"),
                    building_type=row.get("building_type"),
                    building_category=row.get("building_category"),
                    year_built=None if pd.isna(row.get("year_built")) else int(row.get("year_built")),
                    num_dwellings=None if pd.isna(row.get("num_dwellings")) else int(row.get("num_dwellings")),
                    stories=None if pd.isna(row.get("stories")) else int(row.get("stories")),
                    footprint_area=None if pd.isna(row.get("footprint_area")) else float(row.get("footprint_area")),
                    total_floor_area=None if pd.isna(row.get("total_floor_area")) else float(row.get("total_floor_area")),
                    volume=None if pd.isna(row.get("volume")) else float(row.get("volume")),
                    height_m=None if pd.isna(row.get("height_m")) else float(row.get("height_m")),
                    aliases=aliases,
                    provenance=provenance,
                    metadata=metadata,
                )
                city.add_building(building)

            if assign_building_units:
                city.assign_building_units()
        cities[city_name] = city

    return cities
