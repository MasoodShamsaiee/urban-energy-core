# Data Contracts

## Purpose

This package assumes a small number of stable table shapes across loaders and workflow entry points. The goal of this document is to make those assumptions explicit so they can be tested, versioned, and migrated safely.

## Electricity wide tables

Expected shape:

- index: datetime-like local timestamps
- columns: area codes as strings such as FSA codes or DA codes
- values: electricity consumption values

Operational expectations:

- timestamps should be sortable and unique enough for time-series processing
- column names should align with census and geometry identifiers for the same spatial unit
- values should be numeric or coercible to numeric

## Weather table

Required columns:

- `date_time_local`
- `temperature`

Operational expectations:

- `date_time_local` must be parseable as datetimes
- `temperature` must be numeric or coercible to numeric

## Census table

Expected identifier column:

- `GEO UID` or equivalent spatial-unit identifier used as the index in downstream workflows
- DA census loaders also normalize common aliases like `DAUID`, `DAUID21`, and `GEO UID` back to `DAUID`

Operational expectations:

- identifier values should be string-compatible
- population fields used for per-capita processing should be numeric or coercible

## Geometry tables

Expected content:

- per-city geometry tables with an FSA identifier column such as `FSA`, `CFSAUID`, `CFSAUID21`, `GEO_UID`, `GEO UID`, or `CP3`
- optional per-city DA geometry tables with a DA identifier column such as `DAUID`, `DAUID21`, `DA`, `DisseminationArea`, `GEO_UID`, or `GEO UID`
- valid polygon or multipolygon geometry

Operational expectations:

- geometry loaders normalize common FSA aliases back to `FSA`
- geometry loaders normalize common DA aliases back to `DAUID`
- very large GeoJSON files may require loader retries with relaxed GDAL object-size limits

## City object contract

A `City` is expected to carry:

- `name`
- `crs`
- `weather`
- `fsas`
- `das`
- `buildings`

Each `FSA` is expected to carry some or all of:

- `code`
- `geometry`
- `electricity`
- `census`

Each `DA` is expected to carry the same core fields as `FSA`, plus:

- `nearest_fsas`

Each `Building` is expected to carry some or all of:

- `code`
- `geometry`
- `electricity`
- `census`
- `fsa_code`
- `da_code`
- `aliases`
- `provenance`
- `metadata`

## DA fallback contract

When DA geometry and FSA geometry are both present in a city, DA objects may store fallback FSA codes in `nearest_fsas`.

Operational expectations:

- DAs belong directly to a `City`, not under a single FSA
- DA-to-FSA fallback is proximity-based and not a containment guarantee
- fallback ordering is nearest-first

## Building linkage contract

Buildings are separate entities, not spatial aggregation units.

Operational expectations:

- a building may reference both an `fsa_code` and a `da_code`
- those assignments may come from external data or city-level geometric linking
- containment is preferred before centroid-distance fallback when city geometry is available

## Montreal building-source contract

Current Montreal building workflows assume:

- an inventory-like table, currently `LoD1.parquet`, with `ID_UEV` and broad building attributes
- a geometry-rich Montreal building source, currently the managed `Z:\Public\Montreal 3D data\Mtl_Buildings_Dec2022_KKv1.geojson`

Operational expectations:

- `combine_montreal_building_sources(...)` collapses duplicate `building_id` rows before merge
- if multiple sources provide geometry, the primary geometry source takes precedence
- WKB-like geometry values in tabular sources are decoded when possible before downstream use

## Stability rule

If a workflow changes any of the column requirements above, tests and this document should be updated in the same change.
