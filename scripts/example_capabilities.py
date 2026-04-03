from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name == "pandas":
        print(
            "Missing dependency: pandas.\n"
            "Run this script from the project environment, for example:\n"
            "  conda run -n urban-energy-core python scripts/example_capabilities.py",
            file=sys.stderr,
        )
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from urban_energy_core.io import (
    load_all_fsa_census,
    load_city_fsa_geojsons,
    load_city_weather_csvs,
    load_processed_electricity_wide,
)
from urban_energy_core.pipelines import build_cities_from_data


def _default_data_root() -> Path:
    return REPO_ROOT.parent / "urban-energy-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capability walkthrough for urban-energy-core using a configured data root."
    )
    parser.add_argument("--data-root", default=str(_default_data_root()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()

    elec_path = data_root / "data" / "processed" / "electricity" / "elec_rebuilt_new_weather.parquet"
    census_root = data_root / "data" / "raw" / "census" / "FSA scale"
    geometry_root = data_root / "data" / "raw" / "geometry"
    weather_root = data_root / "data" / "raw" / "weather"

    print("Loading package inputs...")
    elec_df = load_processed_electricity_wide(elec_path)
    census_df = load_all_fsa_census(root_dir=census_root, drop_key_col=False, show_progress=False)
    if "GEO UID" in census_df.columns:
        census_df = census_df.set_index("GEO UID")
    census_df.index = census_df.index.astype(str)

    geo = load_city_fsa_geojsons(geometry_dir=geometry_root, show_progress=False)
    weather = load_city_weather_csvs(weather_dir=weather_root, show_progress=False)

    print("Building city objects...")
    cities = build_cities_from_data(
        elec_df=elec_df,
        city_geojsons=geo,
        city_weather=weather,
        census_df=census_df,
        show_progress=False,
    )

    montreal = cities["montreal"]
    first_fsa_code = montreal.list_fsa_codes()[0]
    first_fsa = montreal.get_fsa(first_fsa_code)

    print("")
    print("=== urban-energy-core capabilities ===")
    print(f"Data root: {data_root}")
    print(f"Cities loaded: {sorted(cities.keys())}")
    print(f"Montreal FSA count: {len(montreal.fsas)}")
    print(f"Example FSA: {first_fsa_code}")
    print(f"Electricity frame shape: {montreal.electricity_frame().shape}")

    print("")
    print("1. Per-FSA operations")
    normalized = first_fsa.normalize_for_weather(montreal.weather)
    per_capita = first_fsa.per_capita_consumption()
    prism = first_fsa.apply_heating_prism(montreal.weather)
    short_term = first_fsa.short_term_metrics(show_progress=False)
    print(f"Normalized series length: {len(normalized)}")
    print(f"Per-capita series mean: {float(per_capita.mean()):.6f}")
    print(f"Heating PRISM change point: {prism['heating_change_point_temp_c']:.2f} C")
    print(f"Short-term daily rows: {len(short_term)}")

    print("")
    print("2. City-level tables")
    prism_table = montreal.compute_prism_table(show_progress=False)
    short_term_table = montreal.compute_short_term_table(show_progress=False)
    print(f"PRISM table shape: {prism_table.shape}")
    print(f"Short-term table shape: {short_term_table.shape}")
    print("Top 5 FSAs by heating slope:")
    print(prism_table[["heating_slope_per_hdd", "r2"]].head(5).to_string())

    print("")
    print("3. Optional city map object")
    map_fig = montreal.plot_map(metric="mean", alpha=0.4, figsize=(7, 4), show=False)
    print(f"Map figure type: {type(map_fig).__name__}")

    print("")
    print("4. Example interpretation")
    strongest = prism_table["heating_slope_per_hdd"].idxmax()
    strongest_row = prism_table.loc[strongest]
    print(
        "Highest heating-sensitive FSA in Montreal: "
        f"{strongest} (slope={strongest_row['heating_slope_per_hdd']:.4f}, r2={strongest_row['r2']:.3f})"
    )

    print("")
    print("This script is a package walkthrough using the configured data root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
