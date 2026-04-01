import pandas as pd


def test_project_root_notebooks_parent():
    from pathlib import Path

    from urban_energy_core.pipelines.core_workflows import project_root

    got = project_root(Path(r"C:\tmp\demo\notebooks"))
    assert got == Path(r"C:\tmp\demo")


def test_clean_weather_tables_coerces_types_and_sorts():
    from urban_energy_core.pipelines.core_workflows import clean_weather_tables

    raw = {
        "montreal": pd.DataFrame(
            {
                "date_time_local": ["2024-01-02 01:00", "bad", "2024-01-01 01:00"],
                "temperature": ["-5.0", "x", "-7.5"],
            }
        )
    }
    out = clean_weather_tables(raw)["montreal"]
    assert list(out.columns) == ["date_time_local", "temperature"]
    assert len(out) == 2
    assert out["date_time_local"].is_monotonic_increasing
