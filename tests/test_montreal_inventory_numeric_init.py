import pandas as pd


def test_load_montreal_inventory_handles_missing_numeric_fields():
    from urban_energy_core.io.load_data import _standardize_montreal_building_columns

    df = pd.DataFrame(
        {
            "ID_UEV": ["B001", "B002"],
            "ANNEE_CONS": [1978, None],
            "LIBELLE_UT": ["Logement", "Bureau"],
        }
    )

    out = _standardize_montreal_building_columns(df)

    assert out.loc[0, "building_id"] == "B001"
    assert int(out.loc[0, "year_built"]) == 1978
    assert pd.isna(out.loc[1, "year_built"])
    assert pd.isna(out.loc[0, "footprint_area"])
