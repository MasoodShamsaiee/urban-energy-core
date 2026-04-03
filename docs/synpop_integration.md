# Synthetic Population Integration

`urban-energy-core` and `synthetic-population-qc` should cooperate through DA-level contracts.

- `urban-energy-core` is responsible for city, FSA, DA, building, weather, and observed-energy context.
- `synthetic-population-qc` is responsible for generating synthetic individuals and households by DA.

The clean integration seam is:

1. Build a city in `urban-energy-core`.
2. Export the DA scope and DA metadata needed for synthetic population generation.
3. Run `synthetic-population-qc` for the same DA scope.
4. Bring synthetic outputs back as DA-level summaries, or later attach them to DA/building workflows.

## What synthetic-population-qc should take from core

Use `build_synpop_da_input_table(city, ...)`.

This emits one row per DA with:

- `city_name`
- `da_code`
- `population_2021`
- `nearest_fsa_1`, `nearest_fsa_2`, `nearest_fsa_3`
- `has_census`
- `has_geometry`
- `city_crs`
- optional `geometry`

Use `build_synpop_city_manifest(city, ...)` when you only need the city-level scope summary and DA code list.

## What core expects back from synthetic-population-qc

The minimal expected synthetic population output is an individual table with:

- one row per synthetic person
- a DA identifier column, typically `area`
- optional household ID column, typically `HID`

Use `summarize_synpop_outputs_by_da(synpop_df, ...)` to standardize the synthetic population result into DA-level counts:

- `da_code`
- `n_individuals_syn`
- optional `n_households_syn`

Use `merge_synpop_summary_to_da_input(...)` to align those outputs with the original DA input table from core.

## Important interpretation note

Synthetic population outputs are not energy outputs.

They should be treated as:

- DA-level socio-demographic enrichment
- household and occupancy structure
- demand-model input context

They should not be confused with observed electricity, PRISM metrics, or weather-normalized load series from `urban-energy-core`.

## Minimal example

```python
from urban_energy_core import (
    build_synpop_city_manifest,
    build_synpop_da_input_table,
    merge_synpop_summary_to_da_input,
    summarize_synpop_outputs_by_da,
)

da_input = build_synpop_da_input_table(montreal_city)
manifest = build_synpop_city_manifest(montreal_city)

# later, after synthetic-population-qc runs:
syn_summary = summarize_synpop_outputs_by_da(synpop_df, da_col="area", household_id_col="HID")
da_joined = merge_synpop_summary_to_da_input(da_input, syn_summary)
```

## Recommended next step

When integration work starts, the first target should be a Montreal DA walkthrough that:

- exports the core DA scope
- runs a sampled synthetic population workflow
- joins synthetic DA counts back to the core DA table
- verifies DA-code alignment end to end
