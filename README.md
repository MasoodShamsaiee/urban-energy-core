# urban-energy-core

`urban-energy-core` is the extractable core package from the larger DSM and synthetic-demand research codebase. It contains the reusable electricity, weather, census, geometry, city-object, normalization, PRISM, short-term, anomaly, imputation, and city-build workflows that can stand alone as a foundation repo.

## What is included

- city, FSA, DA, and building domain objects
- electricity and weather loading helpers
- preprocessing and anomaly treatment workflows
- weather normalization and per-capita utilities
- PRISM and short-term diagnostics
- city build pipelines
- plotting helpers used by the core urban-energy workflow
- Montreal building loaders and walkthrough assets
- a first adapter layer for handing building-ready outputs to `HUB`

## Package layout

```text
src/urban_energy_core/
  config.py
  io/
  domain/
  services/
  pipelines/
  plotting/
```

## Quick start

```powershell
conda create -n urban-energy-core python=3.10 -y
conda run -n urban-energy-core python -m pip install -e .[dev,notebooks]
conda run -n urban-energy-core python -c "import urban_energy_core; print('ok')"
```

## Extraction note

This repo was prepared from a larger local research project by moving the reusable energy-core logic into the `urban_energy_core` namespace and preserving backward compatibility in the source project before extraction.

## Data root

- the package now defaults to the sibling data repo `../urban-energy-data`
- some examples also reference the managed Montreal 3D building geometry on `Z:\Public\Montreal 3D data`
- prefer passing explicit file paths into loaders and pipeline entry points when moving beyond the local research layout
- see [docs/data_contracts.md](docs/data_contracts.md) for the expected table shapes that those paths should resolve to

## Spatial units

- `City` can carry both `fsas` and `das`
- `City` can also carry `buildings`
- `DA` objects live directly under `City` because dissemination areas do not map cleanly to a single FSA
- when both DA and FSA geometry are available, DA objects can keep a nearest-FSA fallback list for data-gap handling
- DA workflows can now be loaded through package helpers as optional inputs, not just attached manually
- `Building` is modeled separately from spatial units and can be linked to both a DA and an FSA

## HUB integration

- `urban-energy-core` now includes a first bridge for the sibling `HUB` platform
- use `build_hub_ready_building_table(...)` to inspect Montreal-ready building inputs for HUB
- use `export_hub_building_geojson(...)` to generate a GeoJSON for `HUB`'s `GeometryFactory("geojson", ...)`
- use `to_hub_city(...)` if the local `HUB` repo is available and you want a direct adapter path
- see [docs/hub_integration.md](docs/hub_integration.md) for the intended architecture and current limitations

## Synthetic Population Integration

- `urban-energy-core` now also defines a DA-level contract for working with the sibling `synthetic-population-qc` package
- use `build_synpop_da_input_table(...)` to export the DA scope and fallback context for a city
- use `build_synpop_city_manifest(...)` for a lightweight DA manifest
- use `summarize_synpop_outputs_by_da(...)` and `merge_synpop_summary_to_da_input(...)` to bring synthetic population outputs back into DA-level core tables
- see [docs/synpop_integration.md](docs/synpop_integration.md) for the intended handoff

## Example assets

- [example_capabilities.py](C:/Users/m_hamsai/OneDrive%20-%20Concordia%20University%20-%20Canada/PhD%20Research%20/Codes%20and%20Projects/urban-energy-core/scripts/example_capabilities.py)
- [inspect_data_root.py](C:/Users/m_hamsai/OneDrive%20-%20Concordia%20University%20-%20Canada/PhD%20Research%20/Codes%20and%20Projects/urban-energy-core/scripts/inspect_data_root.py)
- [export_montreal_hub_sample.py](C:/Users/m_hamsai/OneDrive%20-%20Concordia%20University%20-%20Canada/PhD%20Research%20/Codes%20and%20Projects/urban-energy-core/scripts/export_montreal_hub_sample.py)
- [capabilities_walkthrough.ipynb](C:/Users/m_hamsai/OneDrive%20-%20Concordia%20University%20-%20Canada/PhD%20Research%20/Codes%20and%20Projects/urban-energy-core/notebooks/capabilities_walkthrough.ipynb)
- [montreal_city_walkthrough.ipynb](C:/Users/m_hamsai/OneDrive%20-%20Concordia%20University%20-%20Canada/PhD%20Research%20/Codes%20and%20Projects/urban-energy-core/notebooks/montreal_city_walkthrough.ipynb)

## Citation

If you use this repository in research, software, or derivative work, please preserve the license notices and cite the project using [CITATION.cff](CITATION.cff).

## Remaining gaps

- run the full test suite in a fully provisioned env and tighten any remaining environment-specific failures
- continue hardening cross-repo contracts as `syn pop` and later repos integrate
- add a proper demand-disaggregation layer before treating area-observed energy as building-level simulation input
