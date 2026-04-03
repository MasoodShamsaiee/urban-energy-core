def test_package_imports():
    import urban_energy_core
    from urban_energy_core import (
        Building,
        City,
        DA,
        EnergyEntity,
        FSA,
        SpatialUnit,
        build_hub_ready_building_table,
        build_cities_from_data,
        combine_montreal_building_sources,
        default_hub_repo_root,
        export_hub_building_geojson,
        load_montreal_building_geometry,
        load_montreal_building_inventory,
        plot_spatial_samples_with_basemap,
        to_hub_city,
    )

    assert hasattr(urban_energy_core, "load_all_fsa_census")
    assert Building.__name__ == "Building"
    assert City.__name__ == "City"
    assert DA.__name__ == "DA"
    assert EnergyEntity.__name__ == "EnergyEntity"
    assert FSA.__name__ == "FSA"
    assert SpatialUnit.__name__ == "SpatialUnit"
    assert callable(build_hub_ready_building_table)
    assert callable(build_cities_from_data)
    assert callable(load_montreal_building_inventory)
    assert callable(load_montreal_building_geometry)
    assert callable(combine_montreal_building_sources)
    assert callable(default_hub_repo_root)
    assert callable(export_hub_building_geojson)
    assert callable(to_hub_city)
    assert callable(plot_spatial_samples_with_basemap)
