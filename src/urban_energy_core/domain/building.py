from __future__ import annotations

from dataclasses import dataclass, field

from urban_energy_core.domain.energy_entity import EnergyEntity


@dataclass
class Building(EnergyEntity):
    fsa_code: str | None = None
    da_code: str | None = None
    building_type: str | None = None
    building_category: str | None = None
    year_built: int | None = None
    num_dwellings: int | None = None
    stories: int | None = None
    footprint_area: float | None = None
    total_floor_area: float | None = None
    volume: float | None = None
    height_m: float | None = None
    aliases: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
