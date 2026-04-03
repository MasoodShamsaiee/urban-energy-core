import pandas as pd


class _FakePoint:
    def __init__(self, x: float):
        self.x = float(x)

    @property
    def centroid(self):
        return self

    def distance(self, other) -> float:
        return abs(self.x - other.x)


def test_build_cities_can_attach_da_geometry_without_da_electricity():
    from urban_energy_core import build_cities_from_data

    elec_df = pd.DataFrame({"H1A": [1.0, 2.0]})
    fsa_gdf = pd.DataFrame({"FSA": ["H1A"], "geometry": [_FakePoint(0.0)]})
    da_gdf = pd.DataFrame({"DAUID": ["DA001"], "geometry": [_FakePoint(1.0)]})

    cities = build_cities_from_data(
        elec_df=elec_df,
        city_geojsons={"montreal": fsa_gdf},
        city_da_geojsons={"montreal": da_gdf},
        show_progress=False,
    )

    city = cities["montreal"]
    assert city.list_da_codes() == ["DA001"]
    assert city.get_da("DA001").electricity is None
