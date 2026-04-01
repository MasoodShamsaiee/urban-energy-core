# Data Contracts

## Purpose

This package assumes a small number of stable table shapes across loaders and workflow entry points. The goal of this document is to make those assumptions explicit so they can be tested, versioned, and migrated safely.

## Electricity wide table

Expected shape:

- index: datetime-like local timestamps
- columns: FSA codes as strings
- values: electricity consumption values

Operational expectations:

- timestamps should be sortable and unique enough for time-series processing
- column names should align with census and geometry FSA identifiers
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

- `GEO UID` or equivalent FSA identifier used as the index in downstream workflows

Operational expectations:

- identifier values should be string-compatible
- population fields used for per-capita processing should be numeric or coercible

## Geometry tables

Expected content:

- per-city geometry tables with an FSA identifier column such as `FSA`, `CFSAUID`, `CFSAUID21`, `GEO_UID`, `GEO UID`, or `CP3`
- valid polygon or multipolygon geometry

## City object contract

A `City` is expected to carry:

- `name`
- `weather`
- `fsas`

Each `FSA` is expected to carry some or all of:

- `code`
- `geometry`
- `electricity`
- `census`

## Stability rule

If a workflow changes any of the column requirements above, tests and this document should be updated in the same change.
