from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from urban_energy_core.domain.city import City
from urban_energy_core.pipelines.core_workflows import project_root


def default_hub_repo_root(cwd: Path | None = None) -> Path:
    return project_root(cwd).parent / "HUB" / "hub"


def _sample_frame(
    df: pd.DataFrame,
    *,
    sample_frac: float | None = None,
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return df
    if sample_frac is not None and sample_n is not None:
        raise ValueError("Provide only one of sample_frac or sample_n.")
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")
        n = max(1, int(round(len(df) * sample_frac)))
        return df.sample(n=min(n, len(df)), random_state=random_state).sort_index()
    if sample_n is not None:
        if sample_n <= 0:
            raise ValueError("sample_n must be positive.")
        return df.sample(n=min(sample_n, len(df)), random_state=random_state).sort_index()
    return df


def _maybe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _maybe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_maybe_jsonable(v) for v in value]
    if pd.isna(value):
        return None
    return str(value)


def _normalize_mapping(mapping: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(mapping, dict):
        return {}
    return {str(k): _maybe_jsonable(v) for k, v in mapping.items()}


def build_hub_ready_building_table(
    city: City,
    *,
    sample_frac: float | None = None,
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for building in city.buildings.values():
        aliases = _normalize_mapping(building.aliases)
        provenance = _normalize_mapping(building.provenance)
        metadata = _normalize_mapping(building.metadata)

        raw_function = metadata.get("CODE_UTILI")
        if raw_function is None:
            raw_function = metadata.get("function_code")
        if raw_function is None:
            raw_function = building.building_type

        energy_system_archetype = metadata.get("energy_system_archetype")
        if energy_system_archetype is None:
            energy_system_archetype = metadata.get("systems_archetype_name")

        storey_height_m = None
        if building.height_m is not None and building.stories not in (None, 0):
            storey_height_m = float(building.height_m) / float(building.stories)

        fallback_unit_type = None
        fallback_unit_code = None
        if building.da_code is not None:
            fallback_unit_type = "da"
            fallback_unit_code = building.da_code
        elif building.fsa_code is not None:
            fallback_unit_type = "fsa"
            fallback_unit_code = building.fsa_code

        rows.append(
            {
                "id": building.code,
                "building_id": building.code,
                "geometry": building.geometry,
                "year_of_construction": building.year_built,
                "function": raw_function,
                "building_type": building.building_type,
                "building_category": building.building_category,
                "storeys_above_ground": building.stories,
                "storey_height_m": storey_height_m,
                "height_m": building.height_m,
                "footprint_area": building.footprint_area,
                "total_floor_area": building.total_floor_area,
                "volume": building.volume,
                "num_dwellings": building.num_dwellings,
                "energy_system_archetype": energy_system_archetype,
                "fsa_code": building.fsa_code,
                "da_code": building.da_code,
                "fallback_unit_type": fallback_unit_type,
                "fallback_unit_code": fallback_unit_code,
                "energy_data_level": "building" if building.electricity is not None else "area_or_metadata_only",
                "city_name": city.name,
                "city_crs": city.crs,
                "aliases_json": json.dumps(aliases, ensure_ascii=True, sort_keys=True) if aliases else None,
                "provenance_json": json.dumps(provenance, ensure_ascii=True, sort_keys=True) if provenance else None,
                "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True) if metadata else None,
                "hub_ready_notes": (
                    "Geometry and metadata are HUB-ready; observed electricity remains mostly area-level unless "
                    "building electricity is attached."
                ),
                "hub_city_crs": city.crs or "EPSG:2950",
            }
        )

        for alias_key, alias_value in aliases.items():
            rows[-1][f"alias_{alias_key}"] = alias_value

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return _sample_frame(table, sample_frac=sample_frac, sample_n=sample_n, random_state=random_state).reset_index(drop=True)


def export_hub_building_geojson(
    city: City,
    path: str | Path,
    *,
    sample_frac: float | None = None,
    sample_n: int | None = None,
    random_state: int = 42,
    to_wgs84: bool = True,
) -> Path:
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - environment-dependent import
        raise ImportError("GeoPandas is required to export HUB-ready GeoJSON.") from exc

    table = build_hub_ready_building_table(
        city,
        sample_frac=sample_frac,
        sample_n=sample_n,
        random_state=random_state,
    )
    if table.empty:
        raise ValueError(f"City '{city.name}' has no buildings to export.")
    if "geometry" not in table.columns:
        raise ValueError("HUB export requires a geometry column.")

    gdf = gpd.GeoDataFrame(table.copy(), geometry="geometry", crs=city.crs)
    if gdf.crs is None and to_wgs84:
        raise ValueError(
            f"City '{city.name}' has no CRS set, so GeoJSON cannot be safely reprojected for HUB."
        )
    if to_wgs84:
        gdf = gdf.to_crs("EPSG:4326")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def _import_hub_geometry_factory(hub_repo_root: str | Path | None = None):
    if hub_repo_root is not None:
        hub_repo_root = Path(hub_repo_root).resolve()
        if str(hub_repo_root) not in sys.path:
            sys.path.insert(0, str(hub_repo_root))
    try:
        from hub.imports.geometry_factory import GeometryFactory
    except Exception as exc:  # pragma: no cover - optional integration path
        raise ImportError(
            "Could not import HUB's GeometryFactory. Pass hub_repo_root or install HUB in the active environment."
        ) from exc
    return GeometryFactory


def to_hub_city(
    city: City,
    *,
    hub_repo_root: str | Path | None = None,
    output_geojson_path: str | Path | None = None,
    hub_crs: str | None = None,
    sample_frac: float | None = None,
    sample_n: int | None = None,
    random_state: int = 42,
    aliases_fields: list[str] | None = None,
):
    geometry_factory_cls = _import_hub_geometry_factory(
        hub_repo_root=hub_repo_root or default_hub_repo_root()
    )

    if output_geojson_path is None:
        with tempfile.NamedTemporaryFile(prefix=f"{city.name}_hub_", suffix=".geojson", delete=False) as tmp:
            output_geojson_path = Path(tmp.name)
    else:
        output_geojson_path = Path(output_geojson_path)

    export_hub_building_geojson(
        city,
        output_geojson_path,
        sample_frac=sample_frac,
        sample_n=sample_n,
        random_state=random_state,
        to_wgs84=True,
    )

    table = build_hub_ready_building_table(
        city,
        sample_frac=sample_frac,
        sample_n=sample_n,
        random_state=random_state,
    )
    alias_fields = aliases_fields or sorted(c for c in table.columns if c.startswith("alias_"))
    if not alias_fields:
        alias_fields = None

    height_field = "height_m" if "height_m" in table.columns else None
    year_field = "year_of_construction" if "year_of_construction" in table.columns else None
    function_field = "function" if "function" in table.columns else None
    storey_height_field = "storey_height_m" if "storey_height_m" in table.columns else None
    energy_system_archetype_field = (
        "energy_system_archetype" if "energy_system_archetype" in table.columns else None
    )

    return geometry_factory_cls(
        "geojson",
        path=output_geojson_path,
        aliases_field=alias_fields,
        height_field=height_field,
        year_of_construction_field=year_field,
        function_field=function_field,
        storey_height_field=storey_height_field,
        energy_system_archetype_field=energy_system_archetype_field,
        hub_crs=hub_crs or city.crs or "EPSG:2950",
    ).city
