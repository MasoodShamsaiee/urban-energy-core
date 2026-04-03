from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import geopandas as gpd
    import pandas as pd
except ModuleNotFoundError as exc:
    print(
        "Missing dependency.\n"
        "Run this script from the project environment, for example:\n"
        "  conda run -n urban-energy-core python scripts/inspect_data_root.py",
        file=sys.stderr,
    )
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from urban_energy_core.io import load_processed_electricity_wide


def _default_data_root() -> Path:
    return REPO_ROOT.parent / "urban-energy-data"


def _fmt_size(path: Path) -> str:
    size = float(path.stat().st_size)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} B"


def _feature_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(len(gpd.read_file(path, rows=1_000_000)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the current urban-energy-core data root.")
    parser.add_argument("--data-root", default=str(_default_data_root()))
    parser.add_argument("--z-buildings-root", default=r"Z:\Public\Montreal 3D data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    z_root = Path(args.z_buildings_root)

    raw_root = data_root / "data" / "raw"
    processed_root = data_root / "data" / "processed"
    geometry_root = raw_root / "geometry"
    buildings_root = raw_root / "buildings" / "montreal"

    print(f"Data root: {data_root}")
    print("")

    checks = [
        ("FSA electricity", processed_root / "electricity" / "elec_rebuilt_new_weather.parquet"),
        ("Raw electricity", raw_root / "electricity" / "Donnees_4villes_RES.parquet"),
        ("FSA census", raw_root / "census" / "FSA scale"),
        ("DA census", raw_root / "census" / "DA scale"),
        ("Montreal FSA geometry", geometry_root / "Montreal.geojson"),
        ("Montreal DA geometry", geometry_root / "Montreal_DA.geojson"),
        ("Weather Montreal", raw_root / "weather" / "weather_montreal.csv"),
        ("Montreal buildings LoD1", buildings_root / "LoD1.parquet"),
        ("Montreal buildings LoD2 dir", buildings_root / "Montreal_LoD2"),
        ("External Montreal 3D geojson", z_root / "Mtl_Buildings_Dec2022_KKv1.geojson"),
    ]

    for label, path in checks:
        exists = path.exists()
        suffix = _fmt_size(path) if exists and path.is_file() else ("dir" if exists and path.is_dir() else "missing")
        print(f"{label:28} {exists!s:5}  {suffix}")

    print("")
    elec_path = processed_root / "electricity" / "elec_rebuilt_new_weather.parquet"
    if elec_path.exists():
        elec_df = load_processed_electricity_wide(elec_path)
        print(f"Electricity shape: {elec_df.shape}")
        print(f"Montreal-like FSA columns sample: {list(elec_df.columns[:5])}")

    mtl_fsa = geometry_root / "Montreal.geojson"
    if mtl_fsa.exists():
        print(f"Montreal FSA features: {_feature_count(mtl_fsa)}")

    mtl_da = geometry_root / "Montreal_DA.geojson"
    if mtl_da.exists():
        print(f"Montreal DA features: {_feature_count(mtl_da)}")

    lod1 = buildings_root / "LoD1.parquet"
    if lod1.exists():
        lod1_df = pd.read_parquet(lod1)
        print(f"Montreal LoD1 rows: {len(lod1_df)}")

    z_geo = z_root / "Mtl_Buildings_Dec2022_KKv1.geojson"
    if z_geo.exists():
        print(f"External Montreal 3D features: {_feature_count(z_geo)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
