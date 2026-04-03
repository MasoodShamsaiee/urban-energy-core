from pathlib import Path

import pandas as pd


def test_load_processed_da_electricity_wide_delegates_to_processed_loader(monkeypatch):
    from urban_energy_core.io import load_data

    expected = pd.DataFrame({"DA001": [1.0]})
    captured = {}

    def _fake_loader(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(load_data, "load_processed_electricity_wide", _fake_loader)

    out = load_data.load_processed_da_electricity_wide(
        path=Path("tmp") / "da.parquet",
        file_format="parquet",
    )

    assert out is expected
    assert captured["path"] == Path("tmp") / "da.parquet"
    assert captured["kwargs"]["file_format"] == "parquet"


def test_build_city_bundle_from_processed_electricity_can_include_da(monkeypatch):
    from urban_energy_core.pipelines import core_workflows

    idx = pd.date_range("2024-01-01", periods=4, freq="H", tz="America/Toronto")
    fsa_elec = pd.DataFrame({"H1A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    da_elec = pd.DataFrame({"DA001": [0.1, 0.2, 0.3, 0.4]}, index=idx)

    core = core_workflows.CoreProjectData(
        census_df=pd.DataFrame(index=["H1A"]),
        geo={"montreal": object()},
        weather={"montreal": pd.DataFrame({"date_time_local": idx, "temperature": [0.0] * len(idx)})},
        da_census_df=pd.DataFrame(index=["DA001"]),
        da_geo={"montreal": object()},
    )

    class _DummyDA:
        nearest_fsas = ("H1A",)

    class _DummyCity:
        def list_da_codes(self):
            return ["DA001"]

        def get_da(self, code):
            assert code == "DA001"
            return _DummyDA()

    monkeypatch.setattr(core_workflows, "load_core_project_data", lambda **kwargs: core)
    monkeypatch.setattr(core_workflows, "load_processed_electricity_wide", lambda path: fsa_elec)
    monkeypatch.setattr(core_workflows, "load_processed_da_electricity_wide", lambda path=None: da_elec)
    monkeypatch.setattr(
        core_workflows,
        "build_cities_from_data",
        lambda **kwargs: {"montreal": _DummyCity()},
    )

    result = core_workflows.build_city_bundle_from_processed_electricity(
        elec_path="fsa.parquet",
        da_elec_path="da.parquet",
        load_da=True,
        show_progress=False,
    )

    assert result.da_elec_df is da_elec
    assert result.da_census_df is core.da_census_df
    assert result.da_geo is core.da_geo
    assert result.cities["montreal"].list_da_codes() == ["DA001"]
