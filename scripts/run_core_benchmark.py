from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name == "pandas":
        print(
            "Missing dependency: pandas.\n"
            "Run this script from the project environment, for example:\n"
            "  conda run -n urban-energy-core python scripts/run_core_benchmark.py --city montreal\n"
            "Or install the package deps first:\n"
            "  conda run -n urban-energy-core python -m pip install -e .[dev]",
            file=sys.stderr,
        )
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from urban_energy_core.io import (
    load_all_da_census,
    load_all_fsa_census,
    load_city_da_geojsons,
    load_city_fsa_geojsons,
    load_city_weather_csvs,
    load_processed_da_electricity_wide,
    load_processed_electricity_wide,
)
from urban_energy_core.pipelines import build_cities_from_data, rebuild_electricity_with_weather_and_imputation


@dataclass
class StageResult:
    stage: str
    seconds: float
    details: dict[str, Any]


def _time_stage(name: str, fn: Callable[[], Any]) -> tuple[Any, StageResult]:
    print(f"[start] {name}", flush=True)
    t0 = time.perf_counter()
    value = fn()
    dt = time.perf_counter() - t0
    print(f"[done]  {name} ({dt:.2f}s)", flush=True)
    return value, StageResult(stage=name, seconds=dt, details={})


def _default_data_root() -> Path:
    return REPO_ROOT.parent / "urban-energy-data"


def _default_processed_electricity_path(data_root: Path) -> Path:
    return data_root / "data" / "processed" / "electricity" / "elec_rebuilt_new_weather.parquet"


def _default_raw_electricity_path(data_root: Path) -> Path:
    return data_root / "data" / "raw" / "electricity" / "Donnees_4villes_RES.parquet"


def _optional_da_electricity_path(data_root: Path) -> Path | None:
    candidate = data_root / "data" / "raw" / "electricity" / "Donnees_DA.parquet"
    return candidate if candidate.exists() else None


def _find_city_names(city_arg: str, cities: dict[str, Any]) -> list[str]:
    if city_arg == "all":
        return sorted(cities.keys())
    if city_arg not in cities:
        raise KeyError(f"City '{city_arg}' not found. Available: {sorted(cities.keys())}")
    return [city_arg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark urban-energy-core using a configured data root."
    )
    parser.add_argument("--data-root", default=str(_default_data_root()))
    parser.add_argument("--electricity-path", default=None)
    parser.add_argument("--city", default="all", choices=["all", "montreal", "quebec_city", "trois_rivieres"])
    parser.add_argument("--include-da", action="store_true")
    parser.add_argument("--run-rebuild", action="store_true")
    parser.add_argument("--raw-electricity-path", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "benchmarks"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    processed_electricity_path = (
        Path(args.electricity_path).resolve()
        if args.electricity_path is not None
        else _default_processed_electricity_path(data_root)
    )
    if not processed_electricity_path.exists():
        raise FileNotFoundError(f"Processed electricity parquet not found: {processed_electricity_path}")

    raw_root = data_root / "data" / "raw"
    census_fsa_root = raw_root / "census" / "FSA scale"
    census_da_root = raw_root / "census" / "DA scale"
    geometry_root = raw_root / "geometry"
    weather_root = raw_root / "weather"

    stage_results: list[StageResult] = []

    elec_df, stage = _time_stage(
        "load_processed_electricity",
        lambda: load_processed_electricity_wide(processed_electricity_path),
    )
    stage.details = {"shape": list(elec_df.shape), "path": str(processed_electricity_path)}
    stage_results.append(stage)

    census_df, stage = _time_stage(
        "load_fsa_census",
        lambda: load_all_fsa_census(root_dir=census_fsa_root, drop_key_col=False, show_progress=False),
    )
    if "GEO UID" in census_df.columns:
        census_df = census_df.set_index("GEO UID")
    census_df.index = census_df.index.astype(str)
    stage.details = {"shape": list(census_df.shape), "root": str(census_fsa_root)}
    stage_results.append(stage)

    geo, stage = _time_stage(
        "load_fsa_geometry",
        lambda: load_city_fsa_geojsons(geometry_dir=geometry_root, show_progress=False),
    )
    stage.details = {"cities": sorted(geo.keys()), "root": str(geometry_root)}
    stage_results.append(stage)

    weather, stage = _time_stage(
        "load_weather",
        lambda: load_city_weather_csvs(weather_dir=weather_root, show_progress=False),
    )
    stage.details = {"cities": sorted(weather.keys()), "root": str(weather_root)}
    stage_results.append(stage)

    da_elec_df = None
    da_census_df = None
    da_geo = None
    include_da = False
    da_skip_reason = None

    if args.include_da:
        da_path = _optional_da_electricity_path(data_root)
        da_geo_candidates_exist = all(
            (geometry_root / name).exists()
            for name in ("Montreal_DA.geojson", "Quebec_city_DA.geojson", "Trois_Rivieres_DA.geojson")
        )
        if da_path is not None and census_da_root.exists() and da_geo_candidates_exist:
            da_elec_df, stage = _time_stage(
                "load_da_electricity",
                lambda: load_processed_da_electricity_wide(da_path),
            )
            stage.details = {"shape": list(da_elec_df.shape), "path": str(da_path)}
            stage_results.append(stage)

            da_census_df, stage = _time_stage(
                "load_da_census",
                lambda: load_all_da_census(root_dir=census_da_root, drop_key_col=False, show_progress=False),
            )
            if "DAUID" in da_census_df.columns:
                da_census_df = da_census_df.set_index("DAUID")
            da_census_df.index = da_census_df.index.astype(str)
            stage.details = {"shape": list(da_census_df.shape), "root": str(census_da_root)}
            stage_results.append(stage)

            da_geo, stage = _time_stage(
                "load_da_geometry",
                lambda: load_city_da_geojsons(geometry_dir=geometry_root, show_progress=False),
            )
            stage.details = {"cities": sorted(da_geo.keys()), "root": str(geometry_root)}
            stage_results.append(stage)
            include_da = True
        else:
            missing = []
            if da_path is None:
                missing.append("DA electricity parquet")
            if not census_da_root.exists():
                missing.append("DA census folder")
            if not da_geo_candidates_exist:
                missing.append("city DA geojson files")
            da_skip_reason = ", ".join(missing)

    cities, stage = _time_stage(
        "build_cities",
        lambda: build_cities_from_data(
            elec_df=elec_df,
            city_geojsons=geo,
            city_weather=weather,
            census_df=census_df,
            da_elec_df=da_elec_df,
            city_da_geojsons=da_geo,
            da_census_df=da_census_df,
            show_progress=False,
        ),
    )
    stage.details = {
        "cities": sorted(cities.keys()),
        "fsa_counts": {name: len(city.fsas) for name, city in cities.items()},
        "da_counts": {name: len(city.das) for name, city in cities.items()},
    }
    stage_results.append(stage)

    selected_cities = _find_city_names(args.city, cities)
    for city_name in selected_cities:
        city = cities[city_name]

        prism_df, stage = _time_stage(
            f"compute_prism_table:{city_name}",
            lambda city=city: city.compute_prism_table(show_progress=False),
        )
        stage.details = {"shape": list(prism_df.shape)}
        stage_results.append(stage)

        short_df, stage = _time_stage(
            f"compute_short_term_table:{city_name}",
            lambda city=city: city.compute_short_term_table(show_progress=False),
        )
        stage.details = {"shape": list(short_df.shape)}
        stage_results.append(stage)

    rebuild_summary: dict[str, Any] | None = None
    if args.run_rebuild:
        raw_electricity_path = (
            Path(args.raw_electricity_path).resolve()
            if args.raw_electricity_path is not None
            else _default_raw_electricity_path(data_root)
        )
        if not raw_electricity_path.exists():
            raise FileNotFoundError(f"Raw electricity parquet not found: {raw_electricity_path}")

        rebuild_result, stage = _time_stage(
            "rebuild_full_pipeline",
            lambda: rebuild_electricity_with_weather_and_imputation(
                raw_energy_path=raw_electricity_path,
                show_progress=False,
            ),
        )
        stage.details = {
            "raw_shape": list(rebuild_result.elec_raw.shape),
            "rebuilt_shape": list(rebuild_result.elec_rebuilt.shape),
            "imputed_fsas": int(len(rebuild_result.imputation_report)),
        }
        stage_results.append(stage)
        rebuild_summary = stage.details

    summary_rows = [asdict(stage) for stage in stage_results]
    summary_df = pd.DataFrame(
        {
            "stage": [row["stage"] for row in summary_rows],
            "seconds": [row["seconds"] for row in summary_rows],
        }
    ).sort_values("seconds", ascending=False)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"core_benchmark_{ts}.csv"
    json_path = output_dir / f"core_benchmark_{ts}.json"
    summary_df.to_csv(csv_path, index=False)

    payload = {
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "processed_electricity_path": str(processed_electricity_path),
        "selected_cities": selected_cities,
        "include_da_requested": bool(args.include_da),
        "include_da_used": include_da,
        "da_skip_reason": da_skip_reason,
        "run_rebuild": bool(args.run_rebuild),
        "rebuild_summary": rebuild_summary,
        "stages": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Data root: {data_root}")
    print(f"Processed electricity: {processed_electricity_path}")
    if args.include_da and not include_da:
        print(f"DA data skipped: {da_skip_reason}")
    print("")
    print("Stage timings (seconds):")
    print(summary_df.to_string(index=False))
    print("")
    print(f"CSV summary: {csv_path}")
    print(f"JSON summary: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
