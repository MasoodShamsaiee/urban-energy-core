from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from urban_energy_core.domain.building import Building
from urban_energy_core.domain.da import DA
from urban_energy_core.domain.fsa import FSA
from urban_energy_core.domain.spatial_unit import SpatialUnit
from urban_energy_core.services.prism import city_prism_table
from urban_energy_core.services.short_term import city_short_term_table
from urban_energy_core.plotting.city import plot_city_fsa_map


@dataclass
class City:
    name: str
    boundary: Any | None = None
    crs: str | None = None
    weather: pd.DataFrame | None = None
    fsas: dict[str, FSA] = field(default_factory=dict)
    das: dict[str, DA] = field(default_factory=dict)
    buildings: dict[str, Building] = field(default_factory=dict)

    def add_fsa(self, fsa: FSA) -> None:
        self.fsas[fsa.code] = fsa

    def add_da(self, da: DA) -> None:
        self.das[da.code] = da

    def add_building(self, building: Building) -> None:
        self.buildings[building.code] = building

    def set_weather(self, weather_df: pd.DataFrame) -> None:
        self.weather = weather_df

    def list_fsa_codes(self) -> list[str]:
        return sorted(self.fsas.keys())

    def list_da_codes(self) -> list[str]:
        return sorted(self.das.keys())

    def list_building_codes(self) -> list[str]:
        return sorted(self.buildings.keys())

    def get_fsa(self, code: str) -> FSA:
        if code not in self.fsas:
            raise KeyError(f"FSA '{code}' not found in city '{self.name}'.")
        return self.fsas[code]

    def get_da(self, code: str) -> DA:
        if code not in self.das:
            raise KeyError(f"DA '{code}' not found in city '{self.name}'.")
        return self.das[code]

    def get_building(self, code: str) -> Building:
        if code not in self.buildings:
            raise KeyError(f"Building '{code}' not found in city '{self.name}'.")
        return self.buildings[code]

    def _units_for(self, unit: str) -> dict[str, SpatialUnit]:
        if unit == "fsa":
            return self.fsas
        if unit == "da":
            return self.das
        raise ValueError("unit must be one of: 'fsa', 'da'.")

    def _list_codes_for(self, unit: str) -> list[str]:
        if unit == "fsa":
            return self.list_fsa_codes()
        if unit == "da":
            return self.list_da_codes()
        raise ValueError("unit must be one of: 'fsa', 'da'.")

    def electricity_frame(self) -> pd.DataFrame:
        series = []
        for code, fsa in self.fsas.items():
            if fsa.electricity is not None:
                s = fsa.electricity.rename(code)
                series.append(s)
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index()

    def da_electricity_frame(self) -> pd.DataFrame:
        series = []
        for code, da in self.das.items():
            if da.electricity is not None:
                s = da.electricity.rename(code)
                series.append(s)
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index()

    def building_electricity_frame(self) -> pd.DataFrame:
        series = []
        for code, building in self.buildings.items():
            if building.electricity is not None:
                s = building.electricity.rename(code)
                series.append(s)
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index()

    def plot_map(
        self,
        fsas: str | FSA | list[str | FSA] | None = None,
        metric: str = "mean",
        alpha: float = 0.8,
        figsize: tuple[float, float] = (8, 5),
        start=None,
        end=None,
        title: str | None = None,
        show: bool = True,
    ):
        return plot_city_fsa_map(
            city=self,
            fsas=fsas,
            metric=metric,
            alpha=alpha,
            figsize=figsize,
            start=start,
            end=end,
            title=title,
            show=show,
        )

    def normalize_all_fsas_for_weather(
        self,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        in_place: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self._normalize_all_units_for_weather(
            unit="fsa",
            dt_col=dt_col,
            temp_col=temp_col,
            in_place=in_place,
            show_progress=show_progress,
        )

    def normalize_all_das_for_weather(
        self,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        in_place: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self._normalize_all_units_for_weather(
            unit="da",
            dt_col=dt_col,
            temp_col=temp_col,
            in_place=in_place,
            show_progress=show_progress,
        )

    def _normalize_all_units_for_weather(
        self,
        *,
        unit: str,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        in_place: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if self.weather is None:
            raise ValueError(f"City '{self.name}' has no weather dataframe.")

        out = {}
        units = self._units_for(unit)
        codes = self._list_codes_for(unit)
        iter_codes = tqdm(codes, desc=f"Weather-normalizing {unit.upper()}s in {self.name}") if show_progress else codes
        for code in iter_codes:
            area = units[code]
            if area.electricity is None:
                continue
            normalized = area.normalize_for_weather(
                weather_df=self.weather,
                dt_col=dt_col,
                temp_col=temp_col,
                copy=not in_place,
            )
            out[code] = normalized
        if not out:
            return pd.DataFrame()
        return pd.concat(out, axis=1).sort_index()

    def compute_prism_table(
        self,
        unit: str = "fsa",
        per_capita: bool = True,
        weather_normalized: bool = False,
        population_col: str = "Population and dwelling counts / Population, 2021",
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        base_temp_candidates: list[float] | None = None,
        mode: str = "segmented",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return city_prism_table(
            city=self,
            unit=unit,
            per_capita=per_capita,
            weather_normalized=weather_normalized,
            population_col=population_col,
            dt_col=dt_col,
            temp_col=temp_col,
            base_temp_candidates=base_temp_candidates,
            mode=mode,
            show_progress=show_progress,
        )

    def compute_short_term_table(
        self,
        unit: str = "fsa",
        per_capita: bool = True,
        weather_normalized: bool = False,
        population_col: str = "Population and dwelling counts / Population, 2021",
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        am_hours: tuple[int, int] = (6, 11),
        pm_hours: tuple[int, int] = (16, 21),
        dtw_labeler=None,
        use_dtw_clustering: bool = False,
        dtw_k_min: int = 2,
        dtw_k_max: int = 6,
        dtw_min_days: int = 20,
        dtw_dominance_threshold: float = 0.6,
        winter_only: bool = False,
        winter_months: tuple[int, ...] = (11, 12, 1, 2, 3, 4),
        weekday_only: bool = False,
        aggregate: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return city_short_term_table(
            city=self,
            unit=unit,
            per_capita=per_capita,
            weather_normalized=weather_normalized,
            population_col=population_col,
            dt_col=dt_col,
            temp_col=temp_col,
            am_hours=am_hours,
            pm_hours=pm_hours,
            dtw_labeler=dtw_labeler,
            use_dtw_clustering=use_dtw_clustering,
            dtw_k_min=dtw_k_min,
            dtw_k_max=dtw_k_max,
            dtw_min_days=dtw_min_days,
            dtw_dominance_threshold=dtw_dominance_threshold,
            winter_only=winter_only,
            winter_months=winter_months,
            weekday_only=weekday_only,
            aggregate=aggregate,
            show_progress=show_progress,
        )

    def rank_da_to_fsa_distances(self, max_neighbors: int | None = 3) -> dict[str, tuple[str, ...]]:
        if not self.das or not self.fsas:
            return {}

        out: dict[str, tuple[str, ...]] = {}
        for da_code, da in self.das.items():
            if da.geometry is None:
                out[da_code] = tuple()
                self.das[da_code].nearest_fsas = tuple()
                continue

            da_centroid = getattr(da.geometry, "centroid", da.geometry)
            distances: list[tuple[float, str]] = []
            for fsa_code, fsa in self.fsas.items():
                if fsa.geometry is None:
                    continue
                fsa_centroid = getattr(fsa.geometry, "centroid", fsa.geometry)
                try:
                    distance = float(da_centroid.distance(fsa_centroid))
                except Exception:
                    continue
                distances.append((distance, fsa_code))

            distances.sort(key=lambda item: (item[0], item[1]))
            codes = tuple(code for _, code in distances[:max_neighbors]) if max_neighbors is not None else tuple(code for _, code in distances)
            self.das[da_code].nearest_fsas = codes
            out[da_code] = codes

        return out

    def assign_building_units(self, overwrite: bool = False) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {}
        for code, building in self.buildings.items():
            if building.geometry is None:
                out[code] = {"da_code": building.da_code, "fsa_code": building.fsa_code}
                continue

            if overwrite or building.da_code is None:
                building.da_code = self._locate_unit_code(building.geometry, self.das)
            if overwrite or building.fsa_code is None:
                building.fsa_code = self._locate_unit_code(building.geometry, self.fsas)
            out[code] = {"da_code": building.da_code, "fsa_code": building.fsa_code}
        return out

    @staticmethod
    def _locate_unit_code(geometry: Any, units: dict[str, SpatialUnit]) -> str | None:
        centroid = getattr(geometry, "centroid", geometry)
        containing_codes: list[str] = []

        for code, unit in units.items():
            if unit.geometry is None:
                continue
            unit_geometry = unit.geometry

            try:
                if hasattr(unit_geometry, "contains") and unit_geometry.contains(centroid):
                    containing_codes.append(code)
                    continue
            except Exception:
                pass

            try:
                if hasattr(unit_geometry, "intersects") and unit_geometry.intersects(geometry):
                    containing_codes.append(code)
            except Exception:
                continue

        if containing_codes:
            return City._best_code_by_centroid_distance(centroid, units, containing_codes)

        return City._nearest_unit_code_by_centroid_distance(centroid, units)

    @staticmethod
    def _best_code_by_centroid_distance(
        centroid: Any,
        units: dict[str, SpatialUnit],
        codes: list[str],
    ) -> str | None:
        best_code = None
        best_distance = None
        for code in codes:
            unit = units[code]
            if unit.geometry is None:
                continue
            unit_centroid = getattr(unit.geometry, "centroid", unit.geometry)
            try:
                distance = float(centroid.distance(unit_centroid))
            except Exception:
                continue
            if best_distance is None or distance < best_distance or (distance == best_distance and code < best_code):
                best_code = code
                best_distance = distance
        return best_code

    @staticmethod
    def _nearest_unit_code_by_centroid_distance(centroid: Any, units: dict[str, SpatialUnit]) -> str | None:
        return City._best_code_by_centroid_distance(centroid, units, list(units.keys()))
