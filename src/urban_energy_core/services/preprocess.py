import pandas as pd
import numpy as np

MTL_TZ = "America/Toronto"

def preprocess_wide_fsa_timeseries(
    df_wide: pd.DataFrame,
    *,
    tz_local: str = MTL_TZ,
    freq: str | None = None,              # e.g. "H" or "15T"
    min_coverage: float = 0.90,           # drop FSAs with <90% non-missing
    clip_negatives: bool = True,          # set negative to NaN (shouldn't exist)
    fill_method: str | None = None,       # None | "ffill" | "interpolate"
    max_fill_consecutive: int | None = None,  # only if fill_method is set
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess wide electricity data where columns are FSAs and index is time.

    Returns:
      df_clean: cleaned wide dataframe (index = local time, columns = FSAs)
      qc: quality-control table per FSA (coverage, missing count, etc.)
    """
    print("▶ [0] Starting preprocess_wide_fsa_timeseries")
    df = df_wide.copy()

    # --- Stage 1: index checks ---
    print("→ [1] Checking time index")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df_wide index must be a pandas DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df_wide index must be timezone-aware (tz-aware DatetimeIndex)")

    # Convert timezone to Montreal (does not change absolute time, only representation)
    if str(df.index.tz) != tz_local:
        print(f"   • Converting timezone {df.index.tz} → {tz_local}")
        df.index = df.index.tz_convert(tz_local)

    df = df.sort_index()
    print(f"   ✓ Index OK (tz={df.index.tz}, rows={df.shape[0]:,})")

    # --- Stage 2: column checks ---
    print("→ [2] Checking columns (should be FSAs)")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate FSA columns detected")
    # light heuristic: FSAs are 3 chars (letter-digit-letter) in your case
    non_fsa = [c for c in df.columns if not (isinstance(c, str) and len(c) == 3)]
    if non_fsa:
        print(f"   ⚠️  Found non-3char columns (kept anyway): {non_fsa[:10]}{'...' if len(non_fsa)>10 else ''}")
    print(f"   ✓ Columns count: {df.shape[1]}")

    # --- Stage 3: enforce frequency (optional) ---
    if freq is not None:
        print(f"→ [3] Enforcing regular frequency: {freq}")
        before = df.shape[0]
        df = df.asfreq(freq)
        after = df.shape[0]
        introduced = int(df.isna().sum().sum())
        print(f"   ✓ Rows: {before:,} → {after:,}; missing introduced (total cells): {introduced:,}")
    else:
        print("→ [3] Frequency enforcement skipped")

    # --- Stage 4: numeric + negatives ---
    print("→ [4] Coercing to numeric and handling negatives")
    df = df.apply(pd.to_numeric, errors="coerce")

    n_neg = int((df < 0).sum().sum())
    if n_neg > 0:
        msg = f"   ⚠️  Found {n_neg:,} negative values"
        if clip_negatives:
            print(msg + " → setting to NaN")
            df = df.mask(df < 0, np.nan)
        else:
            raise ValueError(msg + " (set clip_negatives=True to auto-fix)")
    else:
        print("   ✓ No negative values")

    # --- Stage 5: QC table before dropping/filling ---
    print("→ [5] Computing QC metrics")
    n_total = df.shape[0]
    na_count = df.isna().sum()
    coverage = 1.0 - (na_count / n_total)
    qc = pd.DataFrame({
        "n_rows": n_total,
        "n_missing": na_count,
        "coverage": coverage,
        "mean_kwh": df.mean(skipna=True),
        "p95_kwh": df.quantile(0.95),
        "max_kwh": df.max(skipna=True),
    }).sort_values("coverage")
    print("   ✓ QC table created")

    # --- Stage 6: drop low-coverage FSAs ---
    print(f"→ [6] Dropping FSAs with coverage < {min_coverage:.0%}")
    keep = qc["coverage"] >= min_coverage
    dropped = qc.index[~keep].tolist()
    df = df.loc[:, keep.values]
    qc = qc.loc[keep]
    print(f"   ✓ Dropped {len(dropped)} FSAs; kept {df.shape[1]} FSAs")

    # --- Stage 7: missing value handling (optional) ---
    print("→ [7] Missing value handling")
    if fill_method is None:
        print("   ✓ No filling (recommended until QC is reviewed)")
    else:
        if fill_method not in {"ffill", "interpolate"}:
            raise ValueError("fill_method must be None, 'ffill', or 'interpolate'")

        if fill_method == "ffill":
            if max_fill_consecutive is None:
                df = df.ffill()
                print("   ✓ Forward-filled all gaps")
            else:
                df = df.ffill(limit=max_fill_consecutive)
                print(f"   ✓ Forward-filled gaps up to {max_fill_consecutive} consecutive steps")

        if fill_method == "interpolate":
            # linear interpolation along time axis
            if max_fill_consecutive is None:
                df = df.interpolate(limit_direction="both")
                print("   ✓ Interpolated all gaps")
            else:
                df = df.interpolate(limit=max_fill_consecutive, limit_direction="both")
                print(f"   ✓ Interpolated gaps up to {max_fill_consecutive} consecutive steps")

    print(f"✔ Done. Final shape: {df.shape[0]:,} rows × {df.shape[1]} FSAs")
    return df, qc
