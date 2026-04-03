import json

import pandas as pd


class _FakePoint:
    def __init__(self, x: float):
        self.x = float(x)

    @property
    def centroid(self):
        return self

    def distance(self, other) -> float:
        return abs(self.x - other.x)


class _FakePolygon:
    def __init__(self, x_min: float, x_max: float, centroid_x: float):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self._centroid = _FakePoint(centroid_x)

    @property
    def centroid(self):
        return self._centroid

    def contains(self, point) -> bool:
        return self.x_min <= point.x <= self.x_max

    def intersects(self, geometry) -> bool:
        point = getattr(geometry, "centroid", geometry)
        return self.contains(point)


def test_assign_building_units_prefers_containment_before_nearest_centroid():
    from urban_energy_core.domain import Building, City, DA, FSA

    city = City(name="montreal")
    city.add_fsa(FSA(code="H1A", geometry=_FakePolygon(0.0, 100.0, 50.0)))
    city.add_fsa(FSA(code="H1B", geometry=_FakePolygon(91.5, 91.7, 91.6)))
    city.add_da(DA(code="DA001", geometry=_FakePolygon(0.0, 100.0, 55.0)))
    city.add_da(DA(code="DA002", geometry=_FakePolygon(91.5, 91.7, 91.6)))
    city.add_building(Building(code="B001", geometry=_FakePoint(90.0), electricity=pd.Series([1.0, 2.0])))

    assigned = city.assign_building_units()

    assert assigned["B001"]["fsa_code"] == "H1A"
    assert assigned["B001"]["da_code"] == "DA001"


def test_build_hub_ready_building_table_carries_bridge_fields():
    from urban_energy_core.domain import Building, City
    from urban_energy_core.integrations import build_hub_ready_building_table

    city = City(name="montreal", crs="EPSG:2950")
    city.add_building(
        Building(
            code="B001",
            geometry=_FakePoint(2.0),
            da_code="DA001",
            fsa_code="H1A",
            building_type="Logement",
            building_category="residential",
            year_built=1978,
            stories=3,
            height_m=12.0,
            num_dwellings=6,
            aliases={"ID_UEV": "01002773"},
            provenance={"inventory": "lod1", "geometry": "mtl_3d"},
            metadata={"CODE_UTILI": "1000", "systems_archetype_name": "montreal_base"},
        )
    )

    table = build_hub_ready_building_table(city)

    row = table.iloc[0]
    assert row["building_id"] == "B001"
    assert row["function"] == "1000"
    assert row["energy_system_archetype"] == "montreal_base"
    assert row["fallback_unit_type"] == "da"
    assert row["fallback_unit_code"] == "DA001"
    assert row["hub_city_crs"] == "EPSG:2950"
    assert row["alias_ID_UEV"] == "01002773"
    assert json.loads(row["provenance_json"])["geometry"] == "mtl_3d"
