import pandas as pd


class _FakePoint:
    def __init__(self, x: float):
        self.x = float(x)

    @property
    def centroid(self):
        return self

    def distance(self, other) -> float:
        return abs(self.x - other.x)


def test_fsa_and_da_are_spatial_units():
    from urban_energy_core.domain import DA, EnergyEntity, FSA, SpatialUnit

    fsa = FSA(code="H1A")
    da = DA(code="DA001")

    assert isinstance(fsa, SpatialUnit)
    assert isinstance(da, SpatialUnit)
    assert isinstance(fsa, EnergyEntity)
    assert isinstance(da, EnergyEntity)


def test_building_is_energy_entity_with_shared_analysis_methods():
    from urban_energy_core.domain import Building, EnergyEntity

    building = Building(code="B001")

    assert isinstance(building, EnergyEntity)
    assert hasattr(building, "normalize_for_weather")
    assert hasattr(building, "apply_prism")
    assert hasattr(building, "short_term_metrics")


def test_city_can_assign_buildings_to_da_and_fsa():
    from urban_energy_core.domain import Building, City, DA, FSA

    city = City(name="montreal")
    city.add_fsa(FSA(code="H1A", geometry=_FakePoint(0.0)))
    city.add_fsa(FSA(code="H1B", geometry=_FakePoint(10.0)))
    city.add_da(DA(code="DA001", geometry=_FakePoint(1.0)))
    city.add_da(DA(code="DA002", geometry=_FakePoint(9.0)))
    city.add_building(
        Building(
            code="B001",
            geometry=_FakePoint(1.5),
            electricity=pd.Series([1.0, 2.0]),
        )
    )

    assigned = city.assign_building_units()

    assert city.list_building_codes() == ["B001"]
    assert assigned["B001"]["da_code"] == "DA001"
    assert assigned["B001"]["fsa_code"] == "H1A"
    assert city.get_building("B001").da_code == "DA001"
    assert city.get_building("B001").fsa_code == "H1A"
