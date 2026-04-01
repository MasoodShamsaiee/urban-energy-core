# urban-energy-core

`urban-energy-core` is the extractable core package from the larger DSM and synthetic-demand research codebase. It contains the reusable electricity, weather, census, geometry, city-object, normalization, PRISM, short-term, anomaly, imputation, and city-build workflows that can stand alone as a foundation repo.

## What is included

- city and FSA domain objects
- electricity and weather loading helpers
- preprocessing and anomaly treatment workflows
- weather normalization and per-capita utilities
- PRISM and short-term diagnostics
- city build pipelines
- plotting helpers used by the core urban-energy workflow

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
conda run -n dsm_qc python -m pip install -e .[dev]
conda run -n dsm_qc python -c "import sys; sys.path.insert(0, 'src'); import urban_energy_core; print('ok')"
```

## Extraction note

This repo was prepared from a larger local research project by moving the reusable energy-core logic into the `urban_energy_core` namespace and preserving backward compatibility in the source project before extraction.

## Path config

- workflow helpers still assume the original research layout by default when they compute the project root
- prefer passing explicit file paths into loaders and pipeline entry points instead of relying on implicit local directories
- see [docs/data_contracts.md](docs/data_contracts.md) for the expected table shapes that those paths should resolve to

## Citation

If you use this repository in research, software, or derivative work, please preserve the license notices and cite the project using [CITATION.cff](CITATION.cff).

## Next cleanup items

- add focused unit tests for loaders, city building, PRISM, and normalization
- trim any remaining research-specific assumptions in config and path handling
- add example notebook/script usage
