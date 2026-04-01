def test_package_imports():
    import urban_energy_core
    from urban_energy_core import City, FSA, build_cities_from_data

    assert hasattr(urban_energy_core, "load_all_fsa_census")
    assert City.__name__ == "City"
    assert FSA.__name__ == "FSA"
    assert callable(build_cities_from_data)
