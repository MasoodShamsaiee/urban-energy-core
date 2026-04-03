from __future__ import annotations

from dataclasses import dataclass, field

from urban_energy_core.domain.spatial_unit import SpatialUnit


@dataclass
class DA(SpatialUnit):
    nearest_fsas: tuple[str, ...] = field(default_factory=tuple)
