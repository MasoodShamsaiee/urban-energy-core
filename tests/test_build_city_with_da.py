import pandas as pd


class _FakePoint:
    def __init__(self, x: float):
        self.x = float(x)

    @property
    def centroid(self):
        return self

    def distance(self, other) -> float:
        return abs(self.x - other.x)


class _FakeGeoFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return _FakeGeoFrame

    @property
    def unary_union(self):
        return self["geometry"].iloc[0]


def test_build_cities_from_data_attaches_das_and_nearest_fsas():
    from urban_energy_core.pipelines.build_city import build_cities_from_data

    idx = pd.date_range("2024-01-01", periods=4, freq="H", tz="America/Toronto")
    elec_df = pd.DataFrame({"H1A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    da_elec_df = pd.DataFrame({"DA001": [0.5, 0.6, 0.7, 0.8]}, index=idx)

    city_geojsons = {
        "montreal": _FakeGeoFrame(
            {
                "FSA": ["H1A"],
                "geometry": [_FakePoint(0.0)],
            }
        )
    }
    city_da_geojsons = {
        "montreal": _FakeGeoFrame(
            {
                "DAUID": ["DA001"],
                "geometry": [_FakePoint(1.5)],
            }
        )
    }
    weather = {"montreal": pd.DataFrame({"date_time_local": idx, "temperature": [0.0] * len(idx)})}
    census_df = pd.DataFrame(
        {"Population and dwelling counts / Population, 2021": [100]},
        index=pd.Index(["H1A"], name="GEO UID"),
    )
    da_census_df = pd.DataFrame(
        {"Population and dwelling counts / Population, 2021": [40]},
        index=pd.Index(["DA001"], name="DAUID"),
    )

    cities = build_cities_from_data(
        elec_df=elec_df,
        city_geojsons=city_geojsons,
        city_weather=weather,
        census_df=census_df,
        da_elec_df=da_elec_df,
        city_da_geojsons=city_da_geojsons,
        da_census_df=da_census_df,
        show_progress=False,
    )

    city = cities["montreal"]
    assert city.list_fsa_codes() == ["H1A"]
    assert city.list_da_codes() == ["DA001"]
    assert city.get_da("DA001").electricity.equals(da_elec_df["DA001"])
    assert city.get_da("DA001").nearest_fsas == ("H1A",)
