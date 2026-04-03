import pandas as pd


def test_build_synpop_da_input_table_and_manifest():
    from urban_energy_core.domain import City, DA, FSA
    from urban_energy_core.integrations import build_synpop_city_manifest, build_synpop_da_input_table

    city = City(name="montreal", crs="EPSG:2950")
    city.add_fsa(FSA(code="H1A"))
    city.add_da(
        DA(
            code="24010018",
            census={"Population and dwelling counts / Population, 2021": 464},
            nearest_fsas=("H1A",),
        )
    )

    da_table = build_synpop_da_input_table(city)
    manifest = build_synpop_city_manifest(city)

    assert da_table.loc[0, "da_code"] == "24010018"
    assert float(da_table.loc[0, "population_2021"]) == 464.0
    assert da_table.loc[0, "nearest_fsa_1"] == "H1A"
    assert manifest["city_name"] == "montreal"
    assert manifest["n_das"] == 1
    assert manifest["da_codes"] == ["24010018"]


def test_summarize_and_merge_synpop_outputs_by_da():
    from urban_energy_core.integrations import (
        merge_synpop_summary_to_da_input,
        summarize_synpop_outputs_by_da,
    )

    syn = pd.DataFrame(
        {
            "area": ["24010018", "24010018", "24010019"],
            "HID": [1, 1, 2],
        }
    )
    da_input = pd.DataFrame(
        {
            "da_code": ["24010018", "24010019"],
            "population_2021": [464.0, 536.0],
        }
    )

    summary = summarize_synpop_outputs_by_da(syn)
    merged = merge_synpop_summary_to_da_input(da_input, summary)

    assert set(summary.columns) >= {"da_code", "n_individuals_syn", "n_households_syn"}
    assert int(merged.loc[merged["da_code"] == "24010018", "n_individuals_syn"].iloc[0]) == 2
    assert int(merged.loc[merged["da_code"] == "24010018", "n_households_syn"].iloc[0]) == 1
