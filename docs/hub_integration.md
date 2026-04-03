# HUB Integration

`urban-energy-core` and `HUB` solve adjacent but different problems.

- `urban-energy-core` is best used as the observed-data and enrichment layer.
- `HUB` is best used as the building-centric simulation and export layer.

The recommended architecture is:

1. Build and enrich a `City` in `urban-energy-core`.
2. Attach Montreal building geometry, attributes, DA/FSA links, and any available building-level electricity.
3. Export a HUB-ready building GeoJSON from `urban-energy-core`.
4. Import that GeoJSON into `HUB` and continue with construction, usage, weather, and energy-system enrichment there.

## Current bridge helpers

- `build_hub_ready_building_table(city, ...)`
  - returns a building table ready to inspect or export
  - includes geometry, Montreal building metadata, fallback DA/FSA links, and provenance fields
- `export_hub_building_geojson(city, path, ...)`
  - writes a GeoJSON that matches `HUB`'s `GeometryFactory("geojson", ...)` path
  - reprojects to `EPSG:4326` for HUB's geojson importer
- `to_hub_city(city, ...)`
  - optional direct adapter when the `HUB` repo is available locally
  - exports a temporary GeoJSON and calls `hub.imports.geometry_factory.GeometryFactory`

## Important limitations

- Most observed electricity is still FSA-level, not building-level.
- That means the bridge is strong for geometry, metadata, and contextual enrichment, but not yet for defensible building demand simulation inputs.
- Building demand disaggregation should remain an explicit modeling step, not a hidden adapter assumption.

## Minimal Montreal flow

```python
from pathlib import Path
from urban_energy_core import export_hub_building_geojson, to_hub_city

hub_geojson = export_hub_building_geojson(montreal_city, Path("outputs/montreal_hub_buildings.geojson"))
hub_city = to_hub_city(montreal_city, output_geojson_path=hub_geojson)
```

## Recommended next steps

- build a Montreal notebook that exports a 5% sample into HUB
- add a Montreal-specific function mapping for `CODE_UTILI -> HUB function`
- add an explicit disaggregation layer before using area-level electricity as building demand
- keep provenance fields attached so HUB users can distinguish observed vs inferred data
