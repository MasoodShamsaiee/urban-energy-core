from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.urban_energy_core.domain.fsa import FSA
from src.urban_energy_core.services.prism import city_prism_table
from src.urban_energy_core.services.short_term import city_short_term_table
from src.urban_energy_core.plotting.city import plot_city_fsa_map


@dataclass
class City:
    name: str
    boundary: Any | None = None
    weather: pd.DataFrame | None = None
    fsas: dict[str, FSA] = field(default_factory=dict)

    def add_fsa(self, fsa: FSA) -> None:
        self.fsas[fsa.code] = fsa

    def set_weather(self, weather_df: pd.DataFrame) -> None:
        self.weather = weather_df

    def list_fsa_codes(self) -> list[str]:
        return sorted(self.fsas.keys())

    def get_fsa(self, code: str) -> FSA:
        if code not in self.fsas:
            raise KeyError(f"FSA '{code}' not found in city '{self.name}'.")
        return self.fsas[code]

    def electricity_frame(self) -> pd.DataFrame:
        series = []
        for code, fsa in self.fsas.items():
            if fsa.electricity is not None:
                s = fsa.electricity.rename(code)
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
        if self.weather is None:
            raise ValueError(f"City '{self.name}' has no weather dataframe.")

        out = {}
        codes = self.list_fsa_codes()
        iter_codes = tqdm(codes, desc=f"Weather-normalizing {self.name}") if show_progress else codes
        for code in iter_codes:
            fsa = self.fsas[code]
            if fsa.electricity is None:
                continue
            normalized = fsa.normalize_for_weather(
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
