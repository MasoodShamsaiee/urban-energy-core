import pandas as pd


class _FakePoint:
    def __init__(self, x: float):
        self.x = float(x)

    @property
    def centroid(self):
        return self

    def distance(self, other) -> float:
        return abs(self.x - other.x)


def test_combine_montreal_building_sources_prefers_broader_attributes_and_keeps_geometry():
    from urban_energy_core.io import combine_montreal_building_sources

    inventory_df = pd.DataFrame(
        {
            "ID_UEV": ["B001"],
            "ANNEE_CONS": [1978],
            "LIBELLE_UT": ["Logement"],
            "BuildingCategory": ["detached"],
            "TotalFloorArea": [900.0],
        }
    )
    geometry_df = pd.DataFrame(
        {
            "ID_UEV": ["B001"],
            "ETAGE_HORS": [3],
            "NOMBRE_LOG": [6],
            "Volume": [3600.0],
            "Z_Min": [10.0],
            "Z_Max": [22.0],
            "geometry": [_FakePoint(2.0)],
        }
    )

    out = combine_montreal_building_sources(
        inventory_df=inventory_df,
        primary_geometry_gdf=geometry_df,
    )

    row = out.iloc[0]
    assert row["building_id"] == "B001"
    assert int(row["year_built"]) == 1978
    assert row["building_type"] == "Logement"
    assert row["building_category"] == "detached"
    assert int(row["stories"]) == 3
    assert int(row["num_dwellings"]) == 6
    assert float(row["volume"]) == 3600.0
    assert float(row["height_m"]) == 12.0
    assert float(row["total_floor_area"]) == 900.0
    assert row["geometry"].x == 2.0


def test_combine_montreal_building_sources_prefers_primary_geometry_over_inventory_geometry():
    from urban_energy_core.io import combine_montreal_building_sources

    inventory_df = pd.DataFrame(
        {
            "ID_UEV": ["B001"],
            "geometry": [_FakePoint(99.0)],
            "ANNEE_CONS": [1978],
        }
    )
    geometry_df = pd.DataFrame(
        {
            "ID_UEV": ["B001"],
            "geometry": [_FakePoint(2.0)],
        }
    )

    out = combine_montreal_building_sources(
        inventory_df=inventory_df,
        primary_geometry_gdf=geometry_df,
    )

    assert out.iloc[0]["geometry"].x == 2.0


def test_combine_montreal_building_sources_collapses_duplicate_building_ids():
    from urban_energy_core.io import combine_montreal_building_sources

    inventory_df = pd.DataFrame(
        {
            "ID_UEV": ["B001", "B001"],
            "ANNEE_CONS": [1978, None],
            "LIBELLE_UT": [None, "Logement"],
        }
    )
    geometry_df = pd.DataFrame(
        {
            "ID_UEV": ["B001"],
            "geometry": [_FakePoint(2.0)],
        }
    )

    out = combine_montreal_building_sources(
        inventory_df=inventory_df,
        primary_geometry_gdf=geometry_df,
    )

    assert len(out) == 1
    assert out.iloc[0]["building_id"] == "B001"
    assert int(out.iloc[0]["year_built"]) == 1978
    assert out.iloc[0]["building_type"] == "Logement"


def test_build_cities_from_data_can_attach_buildings_from_combined_rows():
    from urban_energy_core import build_cities_from_data

    elec_df = pd.DataFrame({"H1A": [1.0, 2.0]})
    fsa_gdf = pd.DataFrame({"FSA": ["H1A"], "geometry": [_FakePoint(0.0)]})
    da_gdf = pd.DataFrame({"DAUID": ["DA001"], "geometry": [_FakePoint(1.0)]})
    building_df = pd.DataFrame(
        {
            "building_id": ["B001"],
            "geometry": [_FakePoint(1.2)],
            "building_type": ["Logement"],
            "building_category": ["detached"],
            "year_built": [1978],
            "stories": [3],
            "num_dwellings": [4],
            "volume": [3200.0],
            "height_m": [11.0],
        }
    )

    cities = build_cities_from_data(
        elec_df=elec_df,
        city_geojsons={"montreal": fsa_gdf},
        da_elec_df=pd.DataFrame({"DA001": [0.5, 0.7]}),
        city_da_geojsons={"montreal": da_gdf},
        city_building_gdfs={"montreal": building_df},
        show_progress=False,
    )

    building = cities["montreal"].get_building("B001")
    assert building.building_type == "Logement"
    assert building.building_category == "detached"
    assert building.year_built == 1978
    assert building.da_code == "DA001"
    assert building.fsa_code == "H1A"
