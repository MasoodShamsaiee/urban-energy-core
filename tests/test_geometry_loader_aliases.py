import pandas as pd


def test_rename_first_present_normalizes_fsa_and_dauid_columns():
    from urban_energy_core.io.load_data import _rename_first_present

    fsa_df = pd.DataFrame({"CFSAUID": ["H1A"], "value": [1]})
    da_df = pd.DataFrame({"GEO UID": ["24010018"], "value": [1]})

    fsa_out = _rename_first_present(fsa_df, "FSA", ("FSA", "CFSAUID"))
    da_out = _rename_first_present(da_df, "DAUID", ("DAUID", "GEO UID"))

    assert "FSA" in fsa_out.columns
    assert fsa_out.loc[0, "FSA"] == "H1A"
    assert "DAUID" in da_out.columns
    assert da_out.loc[0, "DAUID"] == "24010018"
