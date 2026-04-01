from __future__ import annotations

import copy
import math
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as scistats
from tqdm.auto import tqdm

from typing import TYPE_CHECKING

from src.urban_energy_core.services.normalization import align_weather_to_load

if TYPE_CHECKING:
    from src.urban_energy_core.domain.city import City


class LegacyPrismFitter:
    """
    English port of the original multi-segment PRISM fitting logic.
    """

    def __init__(self, x_temp: np.ndarray, y_load: np.ndarray, enable_4seg: bool = True):
        self.x = np.asarray(x_temp, dtype=float)
        self.y = np.asarray(y_load, dtype=float)
        self.enable_4seg = bool(enable_4seg)
        self.model_ini = "0"

    @staticmethod
    def _linfit(xv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv = xv[mask]
        yv = yv[mask]
        if len(xv) < 2 or float(np.nanstd(xv)) < 1e-9:
            raise ValueError("Not enough variation for linear fit.")
        m, b = np.polyfit(xv, yv, 1)
        return float(m), float(b)

    @staticmethod
    def piecewise_linear_4seg(x, x0, x1, x2, y0, y1, k0, k1, k2):
        k1 = (y0 - y1) / (x0 - x1) if (x0 - x1) != 0 else 0.0
        if x1 > x2:
            if (k1 - k2) != 0:
                t = (k1 * x1 - k2 * x2) / (k1 - k2)
                x1, x2 = t, t
                y1 = k1 * (x1 - x0) + y0
        if k1 > 0:
            k1 = 0.0
        if k2 < 0:
            k2 = 0.0
        return np.piecewise(
            x,
            [x <= x0, (x > x0) & (x <= x1), x >= x2],
            [
                lambda xx: k0 * (xx - x0) + y0,
                lambda xx: (y0 - y1) * (xx - x1) / (x0 - x1 + 1e-20) + y1,
                lambda xx: k2 * (xx - x2) + y1,
                lambda xx: y1,
            ],
        )

    @staticmethod
    def piecewise_linear_3seg(x, x0, x1, x2, y0, y1, k0, k1, k2):
        if x1 > x2 and (k1 - k2) != 0:
            t = (k1 * x1 - k2 * x2) / (k1 - k2)
            x1, x2 = t, t
            y1 = k1 * (x1 - x0) + y0
        return np.piecewise(
            x,
            [x <= x1, x >= x2],
            [
                lambda xx: k1 * (xx - x1) + y1,
                lambda xx: k2 * (xx - x2) + y1,
                lambda xx: y1,
            ],
        )

    @staticmethod
    def piecewise_linear_2seg_ch(x, x0, x1, x2, y0, y1, k0, k1, k2):
        if k1 > 0:
            k1 = 0.0
        return np.piecewise(
            x,
            [x <= x1],
            [lambda xx: k1 * (xx - x1) + y1, lambda xx: y1],
        )

    @staticmethod
    def piecewise_linear_2seg_cl(x, x0, x1, x2, y0, y1, k0, k1, k2):
        if k2 < 0:
            k2 = 0.0
        return np.piecewise(
            x,
            [x >= x2],
            [lambda xx: k2 * (xx - x2) + y1, lambda xx: y1],
        )

    @staticmethod
    def piecewise_linear_3be(x, x0, x1, x2, y0, y1, k0, k1, k2):
        if k1 > 0:
            k1 = 0.0
        return np.piecewise(
            x,
            [x <= x0, (x > x0) & (x <= x1)],
            [
                lambda xx: k0 * (xx - x0) + y0,
                lambda xx: (y0 - y1) * (xx - x1) / (x0 - x1 + 1e-20) + y1,
                lambda xx: y1,
            ],
        )

    @staticmethod
    def test_fisher(sse: list[float], num_params: list[int], n_points: int, conf: float = 0.05) -> np.ndarray:
        dim = len(sse)
        f = np.zeros((dim - 1, dim - 1))
        fc = np.zeros((dim - 1, dim - 1))
        for i in range(dim - 1):
            for j in range(i + 1, dim):
                if sse[j] != np.inf:
                    denom = (sse[j] / max(n_points - num_params[j] + 1, 1e-5))
                    f[i, j - 1] = (sse[i] - sse[j]) / (num_params[j] - num_params[i] + 1e-20) / max(denom, 1e-9)
                fc[i, j - 1] = scistats.f.pdf(conf, num_params[j] - num_params[i], max(n_points - num_params[j] + 1, 1))
        return f > fc

    def _init_param_prism(self, x0=-10.0, x1=10.0, x2=20.0) -> list[float]:
        x, y = self.x, self.y
        defaults = [-10.0, 10.0, 20.0, 72.0, 24.0, 2.4, -2.4, 1.0]
        x0i, x1i, x2i = float(x0), float(x1), float(x2)

        def _safe_mean(mask, fallback):
            vals = y[mask]
            if len(vals) == 0:
                return fallback
            return float(np.nanmean(vals))

        y0i = _safe_mean((x > (x0i - 5)) & (x < (x0i + 5)), defaults[3])
        y1i = _safe_mean((x > x1i) & (x < x2i), defaults[4])

        try:
            k1i, b1i = self._linfit(x[(x > x0i) & (x < x1i)], y[(x > x0i) & (x < x1i)])
        except Exception:
            k1i, b1i = -1.0, y1i + x1i
        if (k1i <= -50) or (k1i > 0):
            k1i = (y1i - y0i) / (x1i - x0i + 1e-20)
            b1i = y1i - k1i * x1i

        y0t = _safe_mean(x < x0i, defaults[3])
        x0t = float(np.nanmean(x[x < x0i])) if np.any(x < x0i) else -25.0
        try:
            k0i, b0i = self._linfit(x[x < x0i], y[x < x0i])
        except Exception:
            k0i, b0i = -1.0, y0i
        if (k0i <= -50) or (k0i > 0):
            k0i = (y0i - y0t) / (x0i - x0t + 1e-20)
            b0i = y0i - k0i * x0i

        y2t = _safe_mean(x > x2i, defaults[3])
        x2t = float(np.nanmean(x[x > x2i])) if np.any(x > x2i) else 25.0
        try:
            k2i, b2i = self._linfit(x[x > x2i], y[x > x2i])
        except Exception:
            k2i, b2i = -1.0, y1i
        if (k2i >= 20) or (k2i < 0):
            k2i = (y2t - y1i) / (x2t - x2i + 1e-20)
            b2i = y1i - k2i * x2i
            if k2i < 0:
                k2i = defaults[7]
                b2i = y1i - k2i * x2i

        x0f = (b1i - b0i) / (k0i - k1i + 1e-20)
        x1f = (y1i - b1i) / (k1i + 1e-20)
        x2f = (b2i - y1i) / (-k2i + 1e-20)

        if (x0f < np.min(x)) or (x0f > np.max(x)):
            x0f = x0i
        if (x1f < x0f) or (x1f > np.max(x)):
            x1f = max(x1i, x0f)
        if (x2f < x1f) or (x2f > np.max(x)):
            x2f = max(x1f, x2i)

        p = [x0f, x1f, x2f, y0i, y1i, k0i, k1i, k2i]
        for i, v in enumerate(p):
            if not np.isfinite(v):
                p[i] = defaults[i]
        return p

    def _get_initial_params(self) -> list[float]:
        configs = {
            0: [-10.0, 10.0, 20.0, 72.0, 24.0, 2.4, -2.4, 1.0],
            10: self._init_param_prism(-10, -8, 20),
            11: self._init_param_prism(-10, 10, 20),
            12: self._init_param_prism(-10, 0, 15),
            13: self._init_param_prism(-10, 10, 20),
            14: self._init_param_prism(-10, 10, 20),
            15: self._init_param_prism(0, 10, 20),
            16: self._init_param_prism(-10, 10, 20),
        }
        x, y = self.x, self.y
        errs = {
            0: np.sum((y - self.piecewise_linear_3seg(x, *configs[0])) ** 2),
            10: np.sum((y - self.piecewise_linear_2seg_ch(x, *configs[10])) ** 2),
            11: np.sum((y - self.piecewise_linear_2seg_ch(x, *configs[11])) ** 2),
            12: np.sum((y - self.piecewise_linear_2seg_cl(x, *configs[12])) ** 2),
            13: np.sum((y - self.piecewise_linear_3seg(x, *configs[13])) ** 2),
            14: np.sum((y - self.piecewise_linear_3be(x, *configs[16])) ** 2),
        }
        if self.enable_4seg:
            errs[14] = np.sum((y - self.piecewise_linear_4seg(x, *configs[14])) ** 2)
            errs[15] = np.sum((y - self.piecewise_linear_4seg(x, *configs[15])) ** 2)
        best_key = min(errs, key=errs.get)
        self.model_ini = str(best_key)
        return configs.get(best_key, configs[0])

    def fit(self) -> dict:
        x, y = self.x, self.y
        p0 = self._get_initial_params()
        bounds = (
            [-np.inf, 2, 15, 0, 0, -np.inf, -np.inf, 0],
            [0, 20, 25, np.inf, np.inf, np.inf, 0, np.inf],
        )
        p0 = list(np.clip(p0, bounds[0], bounds[1]))

        candidates = {}
        valid = {"2cl": False, "2ch": False, "3sg": False, "4sg": False, "3be": False}
        method = "dogbox"

        def _try_fit(name, fn):
            try:
                p, _ = optimize.curve_fit(fn, x, y, p0, bounds=bounds, method=method)
                p = np.round(p, 4)
                sse = float(np.sum((y - fn(x, *p)) ** 2))
                cvrmse = float(np.sqrt(sse / max(len(x) - 3, 1)) / max(np.mean(y), 1e-9))
                candidates[name] = {"p": p, "sse": sse, "cvrmse": cvrmse}
                return True
            except Exception:
                return False

        if _try_fit("2ch", self.piecewise_linear_2seg_ch):
            valid["2ch"] = bool(candidates["2ch"]["p"][6] < 0 and np.sum(x < candidates["2ch"]["p"][1]) > 5)
        if _try_fit("2cl", self.piecewise_linear_2seg_cl):
            valid["2cl"] = bool(candidates["2cl"]["p"][7] > 0 and np.sum(x > candidates["2cl"]["p"][2]) > 5)
        if _try_fit("3sg", self.piecewise_linear_3seg):
            p = candidates["3sg"]["p"]
            valid["3sg"] = bool(p[6] < 0 and p[7] > 0 and np.sum(x < p[1]) > 5 and np.sum(x > p[2]) > 5)
        if _try_fit("3be", self.piecewise_linear_3be):
            p = candidates["3be"]["p"]
            valid["3be"] = bool(p[6] < 0 and p[5] >= 0 and np.sum(x < p[1]) > 5)
        if self.enable_4seg and _try_fit("4sg", self.piecewise_linear_4seg):
            p = candidates["4sg"]["p"]
            valid["4sg"] = bool(p[6] < 0 and p[7] > 0 and p[5] >= 0 and np.sum(x < p[1]) > 5 and np.sum(x > p[2]) > 5)

        sse2 = min(
            candidates["2ch"]["sse"] if valid["2ch"] else np.inf,
            candidates["2cl"]["sse"] if valid["2cl"] else np.inf,
        )
        chosen2 = "2ch" if (candidates.get("2ch", {}).get("sse", np.inf) <= candidates.get("2cl", {}).get("sse", np.inf)) else "2cl"
        if not valid.get(chosen2, False):
            chosen2 = "2ch" if valid["2ch"] else "2cl"
        sse3 = min(
            candidates["3sg"]["sse"] if valid["3sg"] else np.inf,
            candidates["3be"]["sse"] if valid["3be"] else np.inf,
        )
        chosen3 = "3sg" if (candidates.get("3sg", {}).get("sse", np.inf) <= candidates.get("3be", {}).get("sse", np.inf)) else "3be"
        if not valid.get(chosen3, False):
            chosen3 = "3sg" if valid["3sg"] else "3be"
        sse4 = candidates.get("4sg", {}).get("sse", np.inf) if valid["4sg"] else np.inf

        sse_list = [sse2, sse3] + ([sse4] if self.enable_4seg else [])
        num_params = [3, 5] + ([7] if self.enable_4seg else [])

        model = chosen2
        if np.isfinite(sse_list[1]):
            fisher = self.test_fisher(sse_list, num_params, len(x), 0.05)
            if fisher[0, 0]:
                model = chosen3
            if self.enable_4seg and len(sse_list) == 3 and fisher.shape[1] >= 2 and (fisher[0, 1] or fisher[1, 1]):
                model = "4sg"

        if model not in candidates:
            finite = {k: v for k, v in candidates.items() if np.isfinite(v["sse"])}
            if not finite:
                raise RuntimeError("All legacy PRISM fits failed.")
            model = min(finite, key=lambda k: finite[k]["sse"])

        p = candidates[model]["p"]
        return {
            "model": model,
            "modele_ini": self.model_ini,
            "x0": float(p[0]),
            "x1": float(p[1]),
            "x2": float(p[2]),
            "y0": float(p[3]),
            "y1": float(p[4]),
            "k0": float(p[5]),
            "k1": float(p[6]),
            "k2": float(p[7]),
            "sse": float(candidates[model]["sse"]),
            "cvrmse": float(candidates[model]["cvrmse"]),
            "n_points": int(len(x)),
            "all_candidates": copy.deepcopy(candidates),
        }


def _piecewise_linear_2seg_ch(x, x0, x1, x2, y0, y1, k0, k1, k2):
    k1 = min(float(k1), 0.0)
    x = np.asarray(x, dtype=float)
    return np.where(x <= x1, k1 * (x - x1) + y1, y1)


def _piecewise_linear_2seg_cl(x, x0, x1, x2, y0, y1, k0, k1, k2):
    k2 = max(float(k2), 0.0)
    x = np.asarray(x, dtype=float)
    return np.where(x >= x2, k2 * (x - x2) + y1, y1)


def _piecewise_linear_3seg(x, x0, x1, x2, y0, y1, k0, k1, k2):
    x = np.asarray(x, dtype=float)
    x1 = float(x1)
    x2 = float(x2)
    if x2 < x1:
        x2 = x1
    k1 = min(float(k1), 0.0)
    k2 = max(float(k2), 0.0)
    out = np.full_like(x, fill_value=float(y1), dtype=float)
    out[x <= x1] = k1 * (x[x <= x1] - x1) + y1
    out[x >= x2] = k2 * (x[x >= x2] - x2) + y1
    return out


def _initial_prism_params(x: np.ndarray, y: np.ndarray) -> list[float]:
    q35, q65 = np.nanquantile(x, [0.35, 0.65])
    y_med = float(np.nanmedian(y))
    y_hi = float(np.nanquantile(y, 0.9))
    y_lo = float(np.nanquantile(y, 0.1))
    k1 = (y_lo - y_med) / max(q35 - float(np.nanmin(x)), 1e-6)
    k2 = (y_hi - y_med) / max(float(np.nanmax(x)) - q65, 1e-6)
    k1 = min(k1, -0.05)
    k2 = max(k2, 0.05)
    return [
        float(np.nanmin(x) + 0.2 * (np.nanmax(x) - np.nanmin(x))),
        float(q35),
        float(q65),
        y_hi,
        y_med,
        0.2,
        k1,
        k2,
    ]


def _fit_segment_model(x: np.ndarray, y: np.ndarray, model: str, p0: list[float]) -> dict:
    bounds = (
        [-40.0, -35.0, -35.0, -1e6, -1e6, -50.0, -50.0, 0.0],
        [40.0, 35.0, 35.0, 1e6, 1e6, 50.0, 0.0, 50.0],
    )
    fn = {
        "2ch": _piecewise_linear_2seg_ch,
        "2cl": _piecewise_linear_2seg_cl,
        "3seg": _piecewise_linear_3seg,
    }[model]
    p_opt, _ = optimize.curve_fit(
        fn,
        x,
        y,
        p0=np.asarray(p0, dtype=float),
        bounds=bounds,
        method="trf",
        maxfev=30000,
    )
    y_hat = fn(x, *p_opt)
    resid = y - y_hat
    sse = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else 0.0
    dof = max(len(y) - 3, 1)
    cvrmse = float(np.sqrt(sse / dof) / max(np.mean(y), 1e-9))

    x1 = float(p_opt[1])
    x2 = float(p_opt[2])
    seg_counts = {
        "left": int(np.sum(x <= x1)),
        "middle": int(np.sum((x > x1) & (x < x2))),
        "right": int(np.sum(x >= x2)),
    }
    valid = True
    if model == "2ch":
        valid = bool(float(p_opt[6]) < 0 and seg_counts["left"] >= 8)
    elif model == "2cl":
        valid = bool(float(p_opt[7]) > 0 and seg_counts["right"] >= 8)
    elif model == "3seg":
        valid = bool(float(p_opt[6]) < 0 and float(p_opt[7]) > 0 and seg_counts["left"] >= 8 and seg_counts["right"] >= 8)

    return {
        "model": model,
        "params": p_opt.astype(float),
        "y_hat": y_hat.astype(float),
        "residuals": resid.astype(float),
        "sse": sse,
        "r2": r2,
        "cvrmse": cvrmse,
        "segment_counts": seg_counts,
        "valid": valid,
    }


def fit_prism_segmented(
    load_series: pd.Series,
    weather_df: pd.DataFrame,
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
    models: tuple[str, ...] = ("2ch", "2cl", "3seg"),
) -> dict:
    """
    Fit segmented PRISM models and select the best valid model by minimum SSE.

    Models:
    - 2ch : heating + baseload
    - 2cl : cooling + baseload
    - 3seg: heating + baseload + cooling
    """
    joined = align_weather_to_load(
        load_series=load_series,
        weather_df=weather_df,
        dt_col=dt_col,
        temp_col=temp_col,
    )
    joined = joined[[temp_col, "load"]].dropna()
    if len(joined) < 20:
        raise ValueError("Not enough aligned points for segmented PRISM fit.")

    x_raw = joined[temp_col].astype(float).to_numpy()
    y_raw = joined["load"].astype(float).to_numpy()
    idx_raw = joined.index
    valid = np.isfinite(x_raw) & np.isfinite(y_raw)
    x = x_raw[valid]
    y = y_raw[valid]
    idx = idx_raw[valid]
    if len(x) < 20:
        raise ValueError("Not enough finite points for segmented PRISM fit.")
    if float(np.nanmax(x) - np.nanmin(x)) < 1e-6:
        raise ValueError("Temperature variation is too small for segmented PRISM fit.")
    fitter = LegacyPrismFitter(x_temp=x, y_load=y, enable_4seg=True)
    result = fitter.fit()
    # keep compatibility with previous "models" argument.
    allowed = {m.replace("3seg", "3sg") for m in models}
    if result["model"] not in allowed:
        cands = result.get("all_candidates", {})
        cands = {k: v for k, v in cands.items() if k in allowed and np.isfinite(v.get("sse", np.inf))}
        if cands:
            alt_model = min(cands, key=lambda k: cands[k]["sse"])
            p_alt = cands[alt_model]["p"]
            result = {
                **result,
                "model": alt_model,
                "x0": float(p_alt[0]),
                "x1": float(p_alt[1]),
                "x2": float(p_alt[2]),
                "y0": float(p_alt[3]),
                "y1": float(p_alt[4]),
                "k0": float(p_alt[5]),
                "k1": float(p_alt[6]),
                "k2": float(p_alt[7]),
                "sse": float(cands[alt_model]["sse"]),
                "cvrmse": float(cands[alt_model]["cvrmse"]),
            }

    y_hat = predict_prism_segmented(x, result)
    resid = y - y_hat
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - float(np.sum(resid**2)) / ss_tot) if ss_tot > 0 else 0.0

    joined_clean = pd.DataFrame({temp_col: x, "load": y}, index=idx).sort_index()
    out = {
        "model": result["model"],
        "modele_ini": result.get("modele_ini"),
        "n_points": int(len(x)),
        "r2": r2,
        "cvrmse": float(result["cvrmse"]),
        "sse": float(result["sse"]),
        "segment_counts": {
            "left": int(np.sum(x <= result["x1"])),
            "middle": int(np.sum((x > result["x1"]) & (x < result["x2"]))),
            "right": int(np.sum(x >= result["x2"])),
        },
        "x0": float(result["x0"]),
        "x1": float(result["x1"]),
        "x2": float(result["x2"]),
        "y0": float(result["y0"]),
        "y1": float(result["y1"]),
        "k0": float(result["k0"]),
        "k1": float(result["k1"]),
        "k2": float(result["k2"]),
        "mean_load": float(np.mean(y)),
        "mean_temp": float(np.mean(x)),
        "joined": joined_clean,
        "y_hat": y_hat.astype(float),
        "residuals": resid.astype(float),
        "all_candidates": result.get("all_candidates", {}),
    }
    return out


def predict_prism_segmented(temp: np.ndarray | pd.Series, fit_result: dict) -> np.ndarray:
    model = fit_result["model"]
    p = [
        fit_result["x0"],
        fit_result["x1"],
        fit_result["x2"],
        fit_result["y0"],
        fit_result["y1"],
        fit_result["k0"],
        fit_result["k1"],
        fit_result["k2"],
    ]
    x = np.asarray(temp, dtype=float)
    if model in {"2ch"}:
        return _piecewise_linear_2seg_ch(x, *p)
    if model in {"2cl"}:
        return _piecewise_linear_2seg_cl(x, *p)
    if model in {"3sg", "3seg"}:
        return _piecewise_linear_3seg(x, *p)
    if model in {"4sg"}:
        return LegacyPrismFitter.piecewise_linear_4seg(x, *p)
    if model in {"3be"}:
        return LegacyPrismFitter.piecewise_linear_3be(x, *p)
    return _piecewise_linear_3seg(x, *p)


def prism_degree_day_summary(
    load_series: pd.Series,
    weather_df: pd.DataFrame,
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
    base_temp_c: float = 18.0,
) -> dict:
    """
    Lightweight PRISM-style summary on hourly data.
    """
    joined = align_weather_to_load(
        load_series=load_series,
        weather_df=weather_df,
        dt_col=dt_col,
        temp_col=temp_col,
    )
    temp = joined[temp_col].astype(float)
    load = joined["load"].astype(float)

    hdd = (base_temp_c - temp).clip(lower=0.0)
    cdd = (temp - base_temp_c).clip(lower=0.0)

    X = np.column_stack([np.ones(len(joined)), hdd.to_numpy(), cdd.to_numpy()])
    y = load.to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "base_temp_c": float(base_temp_c),
        "n_points": int(len(joined)),
        "baseload_intercept": float(coef[0]),
        "heating_slope_per_hdd": float(coef[1]),
        "cooling_slope_per_cdd": float(coef[2]),
        "r2": r2,
        "mean_load": float(np.mean(y)),
        "mean_temp": float(np.mean(temp)),
    }


def prism_heating_change_point_summary(
    load_series: pd.Series,
    weather_df: pd.DataFrame,
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
    base_temp_candidates: list[float] | None = None,
) -> dict:
    """
    Fit heating-only PRISM across candidate heating balance temperatures and pick best R^2.
    Model: load = baseload + heating_slope * HDD(base_temp).
    """
    joined = align_weather_to_load(
        load_series=load_series,
        weather_df=weather_df,
        dt_col=dt_col,
        temp_col=temp_col,
    )
    if len(joined) < 10:
        raise ValueError("Not enough points for PRISM fit.")

    temp = joined[temp_col].astype(float).to_numpy()
    y = joined["load"].astype(float).to_numpy()
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if base_temp_candidates is None:
        base_temp_candidates = np.arange(8.0, 24.5, 0.5).tolist()
    if len(base_temp_candidates) == 0:
        raise ValueError("base_temp_candidates must not be empty.")

    best = None
    for base_temp_c in base_temp_candidates:
        hdd = np.clip(base_temp_c - temp, 0.0, None)
        X = np.column_stack([np.ones(len(y)), hdd])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        candidate = {
            "heating_change_point_temp_c": float(base_temp_c),
            "baseload_intercept": float(coef[0]),
            "heating_slope_per_hdd": float(coef[1]),
            "r2": r2,
        }
        if best is None or candidate["r2"] > best["r2"]:
            best = candidate

    assert best is not None
    return {
        **best,
        "n_points": int(len(joined)),
        "mean_load": float(np.mean(y)),
        "mean_temp": float(np.mean(temp)),
    }


def city_prism_table(
    city: "City",
    per_capita: bool = True,
    weather_normalized: bool = False,
    population_col: str = "Population and dwelling counts / Population, 2021",
    dt_col: str = "date_time_local",
    temp_col: str = "temperature",
    base_temp_candidates: list[float] | None = None,
    mode: str = "segmented",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute heating PRISM summary for all FSAs in a city.
    """
    if city.weather is None:
        raise ValueError(f"City '{city.name}' has no weather dataframe.")

    records = []
    codes = city.list_fsa_codes()
    iter_codes = tqdm(codes, desc=f"PRISM for {city.name}") if show_progress else codes
    for code in iter_codes:
        fsa = city.get_fsa(code)
        if fsa.electricity is None:
            continue

        s = fsa.electricity
        if weather_normalized:
            s = fsa.normalize_for_weather(city.weather, dt_col=dt_col, temp_col=temp_col, copy=True)
        if per_capita:
            s = fsa.per_capita_consumption(population_col=population_col)

        if mode == "segmented":
            summary = fit_prism_segmented(
                load_series=s,
                weather_df=city.weather,
                dt_col=dt_col,
                temp_col=temp_col,
            )
            # keep compatibility with prior plotting columns
            summary["heating_change_point_temp_c"] = summary["x1"]
            summary["baseload_intercept"] = summary["y1"]
            summary["heating_slope_per_hdd"] = -summary["k1"]
            summary["cooling_slope_per_cdd"] = summary["k2"]
            summary.pop("joined", None)
            summary.pop("y_hat", None)
            summary.pop("residuals", None)
        elif mode == "heating_only":
            summary = prism_heating_change_point_summary(
                load_series=s,
                weather_df=city.weather,
                dt_col=dt_col,
                temp_col=temp_col,
                base_temp_candidates=base_temp_candidates,
            )
        else:
            raise ValueError("mode must be one of: 'segmented', 'heating_only'.")
        summary["fsa"] = code
        records.append(summary)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("fsa").sort_values("heating_slope_per_hdd", ascending=False)
