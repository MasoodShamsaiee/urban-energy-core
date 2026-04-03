import pandas as pd


class _FakePoint:
    def __init__(self, x: float):
        self.x = float(x)

    @property
    def centroid(self):
        return self

    def distance(self, other) -> float:
        return abs(self.x - other.x)


def test_city_prism_table_uses_weather_normalized_series_before_per_capita(monkeypatch):
    from urban_energy_core.domain import City, FSA
    from urban_energy_core.services import prism as prism_service

    idx = pd.date_range("2024-01-01", periods=24, freq="H", tz="America/Toronto")
    city = City(
        name="montreal",
        weather=pd.DataFrame({"date_time_local": idx, "temperature": range(len(idx))}),
    )
    fsa = FSA(
        code="H1A",
        electricity=pd.Series([100.0] * len(idx), index=idx, name="H1A"),
        census={"Population and dwelling counts / Population, 2021": 10},
    )
    city.add_fsa(fsa)

    normalized = pd.Series([50.0] * len(idx), index=idx, name="H1A_norm")
    fsa.normalize_for_weather = lambda *args, **kwargs: normalized.copy()

    captured = {}

    def _fake_fit(load_series, weather_df, dt_col, temp_col, models=("2ch", "2cl", "3seg")):
        captured["load_series"] = load_series.copy()
        return {
            "model": "2ch",
            "modele_ini": "2ch",
            "n_points": len(load_series),
            "r2": 1.0,
            "cvrmse": 0.0,
            "sse": 0.0,
            "segment_counts": {"left": 10, "middle": 4, "right": 10},
            "x0": 0.0,
            "x1": 10.0,
            "x2": 20.0,
            "y0": 0.0,
            "y1": 5.0,
            "k0": 0.0,
            "k1": -1.0,
            "k2": 0.0,
            "mean_load": float(load_series.mean()),
            "mean_temp": 0.0,
            "joined": pd.DataFrame(),
            "y_hat": load_series.to_numpy(),
            "residuals": load_series.to_numpy() * 0.0,
            "all_candidates": {},
        }

    monkeypatch.setattr(prism_service, "fit_prism_segmented", _fake_fit)

    out = city.compute_prism_table(
        weather_normalized=True,
        per_capita=True,
        show_progress=False,
    )

    assert "H1A" in out.index
    assert captured["load_series"].eq(5.0).all()


def test_city_short_term_table_uses_weather_normalized_series_before_per_capita(monkeypatch):
    from urban_energy_core.domain import City, FSA
    from urban_energy_core.services import short_term as short_term_service

    idx = pd.date_range("2024-01-01", periods=48, freq="H", tz="America/Toronto")
    city = City(
        name="montreal",
        weather=pd.DataFrame({"date_time_local": idx, "temperature": range(len(idx))}),
    )
    fsa = FSA(
        code="H1A",
        electricity=pd.Series([100.0] * len(idx), index=idx, name="H1A"),
        census={"Population and dwelling counts / Population, 2021": 20},
    )
    city.add_fsa(fsa)

    normalized = pd.Series([40.0] * len(idx), index=idx, name="H1A_norm")
    fsa.normalize_for_weather = lambda *args, **kwargs: normalized.copy()

    captured = {}

    def _fake_compute_daily_short_term_metrics(load_series, **kwargs):
        captured["load_series"] = load_series.copy()
        days = pd.date_range("2024-01-01", periods=2, freq="D")
        return pd.DataFrame(
            {
                "peak_load": [2.0, 2.0],
                "p90_top10_mean": [2.0, 2.0],
                "am_pm_peak_ratio": [1.0, 1.0],
                "ramp_up_rate": [0.0, 0.0],
                "dtw_cluster_label": [pd.NA, pd.NA],
            },
            index=days,
        )

    monkeypatch.setattr(
        short_term_service,
        "compute_daily_short_term_metrics",
        _fake_compute_daily_short_term_metrics,
    )

    out = city.compute_short_term_table(
        weather_normalized=True,
        per_capita=True,
        show_progress=False,
    )

    assert "H1A" in out.index
    assert captured["load_series"].eq(2.0).all()


def test_city_supports_da_objects_and_nearest_fsa_fallbacks(monkeypatch):
    from urban_energy_core.domain import City, DA, FSA
    from urban_energy_core.services import prism as prism_service

    idx = pd.date_range("2024-01-01", periods=24, freq="H", tz="America/Toronto")
    city = City(
        name="montreal",
        weather=pd.DataFrame({"date_time_local": idx, "temperature": range(len(idx))}),
    )

    city.add_fsa(FSA(code="H1A", geometry=_FakePoint(0.0)))
    city.add_fsa(FSA(code="H1B", geometry=_FakePoint(10.0)))
    city.add_da(
        DA(
            code="DA001",
            geometry=_FakePoint(2.0),
            electricity=pd.Series([30.0] * len(idx), index=idx, name="DA001"),
            census={"Population and dwelling counts / Population, 2021": 10},
        )
    )

    nearest = city.rank_da_to_fsa_distances(max_neighbors=2)
    assert nearest["DA001"] == ("H1A", "H1B")
    assert city.get_da("DA001").nearest_fsas == ("H1A", "H1B")

    def _fake_fit(load_series, weather_df, dt_col, temp_col, models=("2ch", "2cl", "3seg")):
        return {
            "model": "2ch",
            "modele_ini": "2ch",
            "n_points": len(load_series),
            "r2": 1.0,
            "cvrmse": 0.0,
            "sse": 0.0,
            "segment_counts": {"left": 10, "middle": 4, "right": 10},
            "x0": 0.0,
            "x1": 10.0,
            "x2": 20.0,
            "y0": 0.0,
            "y1": 3.0,
            "k0": 0.0,
            "k1": -1.0,
            "k2": 0.0,
            "mean_load": float(load_series.mean()),
            "mean_temp": 0.0,
            "joined": pd.DataFrame(),
            "y_hat": load_series.to_numpy(),
            "residuals": load_series.to_numpy() * 0.0,
            "all_candidates": {},
        }

    monkeypatch.setattr(prism_service, "fit_prism_segmented", _fake_fit)

    out = city.compute_prism_table(unit="da", per_capita=True, show_progress=False)
    assert "DA001" in out.index
