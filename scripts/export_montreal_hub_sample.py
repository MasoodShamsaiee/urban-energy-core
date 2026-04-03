from __future__ import annotations

import argparse
from pathlib import Path

from urban_energy_core import (
    build_cities_from_data,
    combine_montreal_building_sources,
    export_hub_building_geojson,
    load_all_da_census,
    load_all_fsa_census,
    load_city_da_geojsons,
    load_city_fsa_geojsons,
    load_city_weather_csvs,
    load_montreal_building_geometry,
    load_montreal_building_inventory,
    load_processed_electricity_wide,
)
from urban_energy_core.config import default_data_dir


def _default_inventory_path(data_root: Path) -> Path:
    return data_root / "raw" / "buildings" / "montreal" / "LoD1.parquet"


def _default_data_root() -> Path:
    return default_data_dir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Montreal building sample from urban-energy-core to HUB-ready GeoJSON.")
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--inventory-path", type=Path, default=None)
    parser.add_argument("--geometry-path", type=Path, required=True, help="Path to the managed Montreal 3D GeoJSON.")
    parser.add_argument("--sample-frac", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("outputs/hub/montreal_hub_sample.geojson"))
    args = parser.parse_args()

    data_root = args.data_root
    inventory_path = args.inventory_path or _default_inventory_path(data_root)

    elec_df = load_processed_electricity_wide()
    fsa_geo = load_city_fsa_geojsons(show_progress=False)
    da_geo = load_city_da_geojsons(show_progress=False)
    weather = load_city_weather_csvs(show_progress=False)
    census_df = load_all_fsa_census(drop_key_col=False, show_progress=False).set_index("GEO UID")
    da_census_df = load_all_da_census(drop_key_col=False, show_progress=False).set_index("DAUID")

    inventory_df = load_montreal_building_inventory(inventory_path)
    geometry_gdf = load_montreal_building_geometry(args.geometry_path)
    buildings_df = combine_montreal_building_sources(
        inventory_df=inventory_df,
        primary_geometry_gdf=geometry_gdf,
    )

    cities = build_cities_from_data(
        elec_df=elec_df,
        city_geojsons=fsa_geo,
        city_weather=weather,
        census_df=census_df,
        city_da_geojsons=da_geo,
        da_census_df=da_census_df,
        city_building_gdfs={"montreal": buildings_df},
        show_progress=True,
    )
    montreal = cities["montreal"]

    out_path = export_hub_building_geojson(
        montreal,
        args.output,
        sample_frac=args.sample_frac,
    )
    print(f"Wrote HUB-ready Montreal sample to {out_path}")


if __name__ == "__main__":
    main()
