from pathlib import Path


def test_load_all_da_census_accepts_geo_uid_metadata_alias(tmp_path: Path):
    from urban_energy_core.io import load_all_da_census

    folder = tmp_path / "age, sex"
    folder.mkdir(parents=True)

    (folder / "sample.txt").write_text(
        "\n".join(
            [
                "2021 Census Profiles Files / Profile of Census Disseminations Areas",
                "COL0 - GEO UID",
                "COL1 - Population and dwelling counts / Population, 2021",
            ]
        ),
        encoding="utf-8",
    )
    (folder / "sample.csv").write_text(
        "\n".join(
            [
                '"COL0","COL1"',
                '"24010018",464',
                '"24010019",536',
            ]
        ),
        encoding="utf-8",
    )

    out = load_all_da_census(root_dir=tmp_path, drop_key_col=False, show_progress=False)

    assert "DAUID" in out.columns
    assert out.loc[0, "DAUID"] == "24010018"
