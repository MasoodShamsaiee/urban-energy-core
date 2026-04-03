import pandas as pd
from pathlib import Path
import os
import re
from tqdm.auto import tqdm

from urban_energy_core.config import (
    CENSUS_DA_SUBDIR,
    CENSUS_FSA_SUBDIR,
    DA_ELEC_FILE,
    default_data_dir,
    ELEC_4CITIES_FILE,
    ELEC_RAW_SUBDIR,
    GEOMETRY_RAW_SUBDIR,
    LOCAL_TZ,
    MONTREAL_DA_GEOJSON,
    MONTREAL_FSA_GEOJSON,
    MONTREAL_WEATHER_FILE,
    QUEBEC_CITY_DA_GEOJSON,
    QUEBEC_CITY_WEATHER_FILE,
    QUEBEC_CITY_FSA_GEOJSON,
    TROIS_RIVIERES_DA_GEOJSON,
    TROIS_RIVIERES_WEATHER_FILE,
    TROIS_RIVIERES_FSA_GEOJSON,
    WEATHER_RAW_SUBDIR,
)


def _read_geojson_with_retry(gpd, file_path: Path):
    try:
        return gpd.read_file(file_path)
    except Exception as exc:
        message = str(exc)
        if "OGR_GEOJSON_MAX_OBJ_SIZE" not in message and "GeoJSON object too complex/large" not in message:
            raise

        previous_limit = os.environ.get("OGR_GEOJSON_MAX_OBJ_SIZE")
        os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
        try:
            return gpd.read_file(file_path)
        except Exception:
            try:
                return gpd.read_file(file_path, engine="fiona")
            except Exception:
                raise exc
        finally:
            if previous_limit is None:
                os.environ.pop("OGR_GEOJSON_MAX_OBJ_SIZE", None)
            else:
                os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = previous_limit


def _rename_first_present(df: pd.DataFrame, canonical: str, candidates: list[str] | tuple[str, ...]):
    if canonical in df.columns:
        return df
    for candidate in candidates:
        if candidate in df.columns:
            return df.rename(columns={candidate: canonical})
    return df


def load_and_prepare_electricity_4cities(
    path: str | Path | None = None,
    tz_local: str = LOCAL_TZ,
) -> pd.DataFrame:
    """
    Load electricity data and return wide FSA format.

    Supported inputs:
    1) Raw Hydro-format parquet with columns CP3, DateIntervalUTC, kWh.
    2) Already-processed wide parquet/csv-like table with datetime index
       (or a datetime column such as dt_local) and FSA columns.

    Output (WIDE):
      - index: dt_local (tz-aware, America/Toronto by default)
      - columns: FSAs (e.g., H2M, G1B, ...)
      - values: kwh (float)

    Drops (if present):
      - Secteur
      - kwh_mean (kWh_Moyen)
      - kwh_std (kWh_std)
      - pct_intervals (pctIntervals)
      - n_clients (nbClients)
    """
    print("▶ [0] Starting load_and_prepare_electricity_4cities")

    # --- Stage 0: Resolve path ---
    print("→ [0.5] Resolving file path") 
    if path is None:
        path = default_data_dir() / "raw" / ELEC_RAW_SUBDIR / ELEC_4CITIES_FILE
    else:
        path = Path(path)

    # --- Stage 1: Read ---
    print("→ [1] Reading parquet")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)
    print(f"   ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # --- Stage 2: Validate/detect schema ---
    print("→ [2] Validating input schema")
    required = ["CP3", "DateIntervalUTC", "kWh"]
    has_raw_schema = all(c in df.columns for c in required)

    # Support already-processed wide electricity input as a convenience path.
    if not has_raw_schema:
        print("   ↳ Raw schema not found. Trying preprocessed wide schema...")
        wide = df.copy()

        # If datetime is stored as a regular column, promote it to index.
        if not isinstance(wide.index, pd.DatetimeIndex):
            dt_candidates = ["dt_local", "date_time_local", "datetime", "timestamp", "time"]
            dt_col = next((c for c in dt_candidates if c in wide.columns), None)
            if dt_col is not None:
                wide[dt_col] = pd.to_datetime(wide[dt_col], errors="coerce")
                bad_ts = int(wide[dt_col].isna().sum())
                if bad_ts:
                    raise ValueError(f"Found {bad_ts} unparsable timestamps in '{dt_col}'.")
                wide = wide.set_index(dt_col)

        if not isinstance(wide.index, pd.DatetimeIndex):
            raise KeyError(
                "Input is neither raw long format nor preprocessed wide format with a datetime index.\n"
                f"Found columns: {list(df.columns)}"
            )

        wide.index = pd.to_datetime(wide.index, errors="coerce")
        bad_idx = int(wide.index.isna().sum())
        if bad_idx:
            raise ValueError(f"Found {bad_idx} unparsable timestamps in index.")

        # Align with historical loader output: tz-aware local time if possible.
        if wide.index.tz is None:
            try:
                wide.index = wide.index.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                # If localization cannot be inferred robustly, keep naive index.
                pass

        wide.columns = [str(c).strip() for c in wide.columns]
        wide = wide.sort_index()
        wide = wide.apply(pd.to_numeric, errors="coerce")

        print(f"   ✓ Detected preprocessed wide input. Final shape: {wide.shape[0]:,} rows × {wide.shape[1]} cols")
        print(f"   Columns sample: {list(wide.columns[:10])}{' ...' if wide.shape[1] > 10 else ''}")
        return wide

    print("   ✓ Detected raw long schema")

    # --- Stage 3: Rename ---
    print("→ [3] Renaming columns")
    df = df.rename(columns={
        "CP3": "FSA",
        "DateIntervalUTC": "dt_utc",
        "kWh": "kwh",
        "kWh_Moyen": "kwh_mean",
        "kWh_std": "kwh_std",
        "pctIntervals": "pct_intervals",
        "nbClients": "n_clients",
        "Secteur": "sector",
    })
    print("   ✓ Renamed: CP3→FSA, DateIntervalUTC→dt_utc, kWh→kwh")

    # --- Stage 4: Drop unwanted columns (if present) ---
    print("→ [4] Dropping unwanted columns")
    drop_cols = ["sector", "kwh_mean", "kwh_std", "pct_intervals", "n_clients"]
    present_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=present_drop, errors="ignore")
    print(f"   ✓ Dropped: {present_drop if present_drop else 'none'}")

    # --- Stage 5: Cast types ---
    print("→ [5] Casting dtypes (FSA, time, kwh)")
    df["FSA"] = df["FSA"].astype("string").str.strip()

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    bad_ts = int(df["dt_utc"].isna().sum())
    if bad_ts:
        raise ValueError(f"Found {bad_ts} unparsable timestamps in dt_utc")

    # Convert UTC -> Montreal local time
    df["dt_local"] = df["dt_utc"].dt.tz_convert(tz_local)

    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    bad_kwh = int(df["kwh"].isna().sum())
    if bad_kwh:
        raise ValueError(f"Found {bad_kwh} non-numeric kwh values after conversion")

    print("   ✓ Dtypes casted and validated")

    # --- Stage 6: Final selection + index ---
    print("→ [6] Selecting final columns and setting index")
    df = df[["dt_local", "FSA", "kwh"]].set_index("dt_local").sort_index()
    print(f"   ✓ Index set to dt_local (tz={df.index.tz})")

    # --- Stage 7: Widen (pivot) ---
    print("→ [7] Pivoting to wide format (columns = FSA)")
    df = df.pivot_table(values="kwh", index=df.index, columns="FSA", aggfunc="sum")

    print(f"✔ Done. Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Columns sample: {list(df.columns[:10])}{' ...' if df.shape[1] > 10 else ''}")

    return df


def load_all_fsa_census(
    root_dir: str | Path | None = None,
    key_col: str = "GEO UID",
    key_col_candidates: list[str] | tuple[str, ...] | None = None,
    how: str = "outer",
    drop_key_col: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Load all FSA census datasets from subfolders and merge into one DataFrame.

    For each subfolder in root_dir:
    - reads a single CSV
    - reads a single TXT metadata file
    - renames columns using COLx definitions in TXT
    Then merges all DataFrames on `key_col`.
    """
    if root_dir is None:
        root_dir = default_data_dir() / "raw" / CENSUS_FSA_SUBDIR
    else:
        root_dir = Path(root_dir)

    def _load_one_folder(folder: Path) -> pd.DataFrame:
        csv_files = sorted(folder.glob("*.csv"))
        txt_files = sorted(folder.glob("*.txt"))

        if len(csv_files) != 1 or len(txt_files) != 1:
            raise ValueError(
                f"{folder}: expected exactly 1 CSV and 1 TXT, found {len(csv_files)} CSV / {len(txt_files)} TXT"
            )

        csv_path = csv_files[0]
        txt_path = txt_files[0]

        col_pattern = re.compile(r"^\s*COL(\d+)\s*-\s*(.+?)\s*$", re.IGNORECASE)
        idx_to_name: dict[int, str] = {}
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = col_pattern.match(line)
                if match:
                    idx_to_name[int(match.group(1))] = match.group(2).strip()

        if not idx_to_name:
            raise ValueError(f"{txt_path}: no COLx definitions found")

        df = pd.read_csv(csv_path)

        rename_map = {f"COL{i}": name for i, name in idx_to_name.items()}
        if any(c in df.columns for c in rename_map):
            df = df.rename(columns=rename_map)
        else:
            new_cols = list(df.columns)
            for i, name in idx_to_name.items():
                if i < len(new_cols):
                    new_cols[i] = name
            df.columns = new_cols

        return df

    subfolders = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in: {root_dir}")

    candidate_keys = [key_col]
    if key_col_candidates is not None:
        candidate_keys.extend([str(c) for c in key_col_candidates if str(c) not in candidate_keys])

    dfs: list[pd.DataFrame] = []
    folders_iter = tqdm(subfolders, desc="Loading census folders") if show_progress else subfolders
    for folder in folders_iter:
        df = _load_one_folder(folder)
        matched_key = next((candidate for candidate in candidate_keys if candidate in df.columns), None)
        if matched_key is None:
            raise KeyError(
                f"{folder}: none of the key columns {candidate_keys} were found after renaming"
            )
        if matched_key != key_col:
            df = df.rename(columns={matched_key: key_col})
        dfs.append(df)

    merged = dfs[0]
    merge_iter = tqdm(dfs[1:], desc="Merging census tables") if show_progress else dfs[1:]
    for df in merge_iter:
        merged = merged.merge(df, on=key_col, how=how, suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

    if key_col in merged.columns:
        merged[key_col] = merged[key_col].astype(str).str.strip()

    if drop_key_col and key_col in merged.columns:
        merged = merged.drop(columns=key_col)

    return merged


def load_all_da_census(
    root_dir: str | Path | None = None,
    key_col: str = "DAUID",
    how: str = "outer",
    drop_key_col: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    if root_dir is None:
        root_dir = default_data_dir() / "raw" / CENSUS_DA_SUBDIR
    return load_all_fsa_census(
        root_dir=root_dir,
        key_col=key_col,
        key_col_candidates=("DAUID", "DAUID21", "GEO UID", "GEO_UID"),
        how=how,
        drop_key_col=drop_key_col,
        show_progress=show_progress,
    )


def load_city_fsa_geojsons(
    geometry_dir: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, "gpd.GeoDataFrame"]:
    """
    Load FSA geometry GeoJSONs for Montreal, Quebec City, and Trois-Rivieres.

    Returns
    -------
    dict[str, geopandas.GeoDataFrame]
        Keys: "montreal", "quebec_city", "trois_rivieres".
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - environment-dependent import
        raise ImportError(
            "Failed to import geopandas/shapely. This is commonly caused by binary "
            "incompatibility with NumPy in the current environment."
        ) from exc

    if geometry_dir is None:
        geometry_dir = default_data_dir() / "raw" / GEOMETRY_RAW_SUBDIR
    else:
        geometry_dir = Path(geometry_dir)

    city_files = {
        "montreal": MONTREAL_FSA_GEOJSON,
        "quebec_city": QUEBEC_CITY_FSA_GEOJSON,
        "trois_rivieres": TROIS_RIVIERES_FSA_GEOJSON,
    }

    out: dict[str, gpd.GeoDataFrame] = {}
    city_iter = tqdm(city_files.items(), desc="Loading city geojsons") if show_progress else city_files.items()
    for city, filename in city_iter:
        file_path = geometry_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"GeoJSON not found for '{city}': {file_path}")
        gdf = _read_geojson_with_retry(gpd, file_path)
        gdf = _rename_first_present(gdf, "FSA", ("FSA", "CFSAUID", "CFSAUID21", "GEO UID", "GEO_UID", "CP3"))
        out[city] = gdf

    return out


def load_city_da_geojsons(
    geometry_dir: str | Path | None = None,
    strict: bool = False,
    show_progress: bool = True,
) -> dict[str, "gpd.GeoDataFrame"]:
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - environment-dependent import
        raise ImportError(
            "Failed to import geopandas/shapely. This is commonly caused by binary "
            "incompatibility with NumPy in the current environment."
        ) from exc

    if geometry_dir is None:
        geometry_dir = default_data_dir() / "raw" / GEOMETRY_RAW_SUBDIR
    else:
        geometry_dir = Path(geometry_dir)

    city_files = {
        "montreal": MONTREAL_DA_GEOJSON,
        "quebec_city": QUEBEC_CITY_DA_GEOJSON,
        "trois_rivieres": TROIS_RIVIERES_DA_GEOJSON,
    }

    out: dict[str, gpd.GeoDataFrame] = {}
    city_iter = tqdm(city_files.items(), desc="Loading city DA geojsons") if show_progress else city_files.items()
    for city, filename in city_iter:
        file_path = geometry_dir / filename
        if not file_path.exists():
            if strict:
                raise FileNotFoundError(f"DA GeoJSON not found for '{city}': {file_path}")
            continue
        gdf = _read_geojson_with_retry(gpd, file_path)
        gdf = _rename_first_present(gdf, "DAUID", ("DAUID", "DAUID21", "DA", "DisseminationArea", "GEO UID", "GEO_UID"))
        out[city] = gdf

    return out


def load_processed_da_electricity_wide(
    path: str | Path | None = None,
    *,
    file_format: str | None = None,
    index_col: str | None = None,
    utc: bool | None = None,
) -> pd.DataFrame:
    if path is None:
        path = default_data_dir() / "raw" / ELEC_RAW_SUBDIR / DA_ELEC_FILE
    return load_processed_electricity_wide(
        path=path,
        file_format=file_format,
        index_col=index_col,
        utc=utc,
    )


def load_weather_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a weather CSV from a given path and keep only:
    - date_time_local
    - temperature
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.name}")

    df = pd.read_csv(path, **read_csv_kwargs)

    if "date_time_local" not in df.columns:
        raise KeyError(f"'date_time_local' column not found. Available columns: {list(df.columns)}")

    if "temperature" in df.columns:
        temp_col = "temperature"
    else:
        temp_candidates = [c for c in df.columns if "temperature" in str(c).lower()]
        if not temp_candidates:
            raise KeyError(f"'temperature' column not found. Available columns: {list(df.columns)}")
        temp_col = temp_candidates[0]

    out = df[["date_time_local", temp_col]].copy()
    if temp_col != "temperature":
        out = out.rename(columns={temp_col: "temperature"})
    return out


def load_city_weather_csvs(
    weather_dir: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load weather CSVs for Montreal, Quebec City, and Trois-Rivieres.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "montreal", "quebec_city", "trois_rivieres".
    """
    if weather_dir is None:
        weather_dir = default_data_dir() / "raw" / WEATHER_RAW_SUBDIR
    else:
        weather_dir = Path(weather_dir)

    city_files = {
        "montreal": MONTREAL_WEATHER_FILE,
        "quebec_city": QUEBEC_CITY_WEATHER_FILE,
        "trois_rivieres": TROIS_RIVIERES_WEATHER_FILE,
    }

    out: dict[str, pd.DataFrame] = {}
    city_iter = tqdm(city_files.items(), desc="Loading city weather files") if show_progress else city_files.items()
    for city, filename in city_iter:
        out[city] = load_weather_csv(weather_dir / filename)
    return out


def save_processed_electricity_wide(
    elec_df: pd.DataFrame,
    path: str | Path,
    *,
    file_format: str | None = None,
    index_name: str = "dt_local",
) -> Path:
    """
    Save processed wide electricity dataframe for reuse as fresh input.

    Expected format:
    - index: DatetimeIndex
    - columns: FSA codes
    - values: electricity (kWh)
    """
    if not isinstance(elec_df.index, pd.DatetimeIndex):
        raise TypeError("elec_df index must be a DatetimeIndex.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = file_format.lower() if file_format is not None else out_path.suffix.lower().lstrip(".")
    if not fmt:
        fmt = "parquet"
        out_path = out_path.with_suffix(".parquet")

    df = elec_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        raise ValueError("elec_df index contains invalid datetimes.")
    df = df.sort_index()
    df.index.name = index_name

    if fmt == "parquet":
        df.to_parquet(out_path)
    elif fmt == "csv":
        df.to_csv(out_path)
    else:
        raise ValueError("file_format must be one of: 'parquet', 'csv'.")

    return out_path


def load_processed_electricity_wide(
    path: str | Path,
    *,
    file_format: str | None = None,
    index_col: str | None = None,
    utc: bool | None = None,
) -> pd.DataFrame:
    """
    Load processed wide electricity dataframe saved by save_processed_electricity_wide.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Processed electricity file not found: {in_path}")

    fmt = file_format.lower() if file_format is not None else in_path.suffix.lower().lstrip(".")
    if fmt == "parquet":
        df = pd.read_parquet(in_path)
    elif fmt == "csv":
        idx_col = index_col or "dt_local"
        df = pd.read_csv(in_path)
        if idx_col not in df.columns:
            idx_col = df.columns[0]
        df[idx_col] = pd.to_datetime(df[idx_col], errors="coerce", utc=utc)
        bad = int(df[idx_col].isna().sum())
        if bad:
            raise ValueError(f"Found {bad} unparsable timestamps in '{idx_col}'.")
        df = df.set_index(idx_col)
    else:
        raise ValueError("file_format must be one of: 'parquet', 'csv'.")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="raise", utc=utc)
        except Exception as exc:
            raise TypeError("Loaded electricity index is not a valid DatetimeIndex.") from exc

    df.columns = [str(c) for c in df.columns]
    df = df.sort_index()
    return df


def _require_geopandas():
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - environment-dependent import
        raise ImportError(
            "Failed to import geopandas/shapely. This is commonly caused by binary "
            "incompatibility with NumPy in the current environment."
        ) from exc
    return gpd


def _coerce_geometry_series(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series

    sample = non_null.iloc[0]
    if not isinstance(sample, (bytes, bytearray, memoryview, str)):
        return series

    try:
        from shapely import wkb
    except Exception:
        return series

    def _decode(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            try:
                return wkb.loads(value)
            except Exception:
                return value
        if isinstance(value, str):
            try:
                return wkb.loads(bytes.fromhex(value))
            except Exception:
                return value
        return value

    return series.map(_decode)


def _normalize_building_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _first_present(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for col in candidates:
        if col in df.columns:
            out = out.where(out.notna(), df[col])
    return out


def _as_numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(float("nan"), index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), vals)
    return out


def _standardize_montreal_building_columns(df: pd.DataFrame, *, id_col: str = "ID_UEV") -> pd.DataFrame:
    out = df.copy()
    if id_col not in out.columns:
        raise KeyError(f"Building ID column '{id_col}' not found in columns: {list(out.columns)}")

    if "geometry" in out.columns:
        out["geometry"] = _coerce_geometry_series(out["geometry"])

    out[id_col] = _normalize_building_id(out[id_col])
    out["building_id"] = out[id_col]
    out["year_built"] = _as_numeric_series(out, ["ANNEE_CONS"]).astype("Int64")
    out["building_type"] = _first_present(out, ["LIBELLE_UT", "g_usag_pre"]).astype("string")
    out["building_category"] = _first_present(out, ["BuildingCategory", "CATEGORIE_"]).astype("string")
    out["num_dwellings"] = _as_numeric_series(out, ["NOMBRE_LOG"]).astype("Int64")
    out["stories"] = _as_numeric_series(out, ["ETAGE_HORS", "FloorNu_RawTax"]).astype("Int64")
    out["footprint_area"] = _as_numeric_series(out, ["Area", "AREA_NEW", "Shape_Area"])
    out["total_floor_area"] = _as_numeric_series(out, ["TotalFloorArea", "area"])
    out["volume"] = _as_numeric_series(out, ["Volume", "BuildingVolume"])

    z_min = _as_numeric_series(out, ["Z_Min"])
    z_max = _as_numeric_series(out, ["Z_Max"])
    height = z_max - z_min
    height = height.where(height.notna(), _as_numeric_series(out, ["Hieght_LiD", "citygml_me"]))
    out["height_m"] = height
    return out


def _collapse_duplicate_building_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "building_id" not in df.columns or df["building_id"].is_unique:
        return df

    def _first_non_null(series: pd.Series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return series.iloc[0] if len(series) else None
        return non_null.iloc[0]

    aggregations = {col: _first_non_null for col in df.columns if col != "building_id"}
    collapsed = (
        df.groupby("building_id", dropna=False, sort=False)
        .agg(aggregations)
        .reset_index()
    )
    if "ID_UEV" in collapsed.columns:
        collapsed["ID_UEV"] = collapsed["building_id"]
    return collapsed


def load_montreal_building_inventory(
    path: str | Path,
    *,
    id_col: str = "ID_UEV",
) -> pd.DataFrame:
    in_path = Path(path)
    suffix = in_path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(in_path)
    elif suffix == ".csv":
        df = pd.read_csv(in_path)
    elif suffix in {".geojson", ".json", ".shp"}:
        gpd = _require_geopandas()
        df = pd.DataFrame(gpd.read_file(in_path).drop(columns="geometry", errors="ignore"))
    else:
        raise ValueError("Unsupported building inventory format. Expected parquet/csv/geojson/shp.")

    return _standardize_montreal_building_columns(df, id_col=id_col)


def load_montreal_building_geometry(
    path: str | Path,
    *,
    id_col: str = "ID_UEV",
) -> "gpd.GeoDataFrame":
    gpd = _require_geopandas()
    gdf = gpd.read_file(Path(path))
    gdf = _standardize_montreal_building_columns(gdf, id_col=id_col)
    return gdf


def combine_montreal_building_sources(
    *,
    inventory_df: pd.DataFrame | None = None,
    primary_geometry_gdf: "gpd.GeoDataFrame | None" = None,
    secondary_geometry_gdf: "gpd.GeoDataFrame | None" = None,
    id_col: str = "ID_UEV",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if inventory_df is not None:
        inv = _standardize_montreal_building_columns(pd.DataFrame(inventory_df).copy(), id_col=id_col)
        inv = _collapse_duplicate_building_rows(inv)
        inv = inv.set_index("building_id", drop=False)
        frames.append(inv)

    if primary_geometry_gdf is not None:
        pri = _standardize_montreal_building_columns(primary_geometry_gdf.copy(), id_col=id_col)
        pri = _collapse_duplicate_building_rows(pri)
        pri = pri.set_index("building_id", drop=False)
        frames.append(pri)

    if secondary_geometry_gdf is not None:
        sec = _standardize_montreal_building_columns(secondary_geometry_gdf.copy(), id_col=id_col)
        sec = _collapse_duplicate_building_rows(sec)
        sec = sec.set_index("building_id", drop=False)
        frames.append(sec)

    if not frames:
        return pd.DataFrame()

    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.combine_first(frame)

    if primary_geometry_gdf is not None:
        pri = _standardize_montreal_building_columns(primary_geometry_gdf.copy(), id_col=id_col)
        pri = _collapse_duplicate_building_rows(pri)
        pri = pri.set_index("building_id", drop=False)
        if "geometry" in pri.columns:
            existing_geometry = merged["geometry"] if "geometry" in merged.columns else None
            merged["geometry"] = pri["geometry"].combine_first(existing_geometry)

    if secondary_geometry_gdf is not None:
        sec = _standardize_montreal_building_columns(secondary_geometry_gdf.copy(), id_col=id_col)
        sec = _collapse_duplicate_building_rows(sec)
        sec = sec.set_index("building_id", drop=False)
        if "geometry" in sec.columns:
            existing_geometry = merged["geometry"] if "geometry" in merged.columns else None
            merged["geometry"] = existing_geometry.combine_first(sec["geometry"]) if existing_geometry is not None else sec["geometry"]

    merged.index.name = "building_id"
    if "building_id" not in merged.columns:
        merged["building_id"] = merged.index.astype(str)
    return merged.reset_index(drop=True)
