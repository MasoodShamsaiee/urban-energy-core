from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from urban_energy_core.services.normalization import (
    compute_per_capita_series,
    normalize_fsa_weather_linear,
)
from urban_energy_core.services.prism import (
    fit_prism_segmented,
    prism_heating_change_point_summary,
)
from urban_energy_core.services.short_term import (
    cluster_daily_profiles_dtw,
    compute_daily_short_term_metrics,
)
from urban_energy_core.plotting.diagnostics import plot_fsa_prism_fit


@dataclass
class FSA:
    code: str
    geometry: Any | None = None
    electricity: pd.Series | None = None
    census: pd.Series | dict | None = None

    def normalize_for_weather(
        self,
        weather_df: pd.DataFrame,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        copy: bool = True,
    ) -> pd.Series:
        if self.electricity is None:
            raise ValueError(f"FSA '{self.code}' has no electricity series.")
        normalized = normalize_fsa_weather_linear(
            load_series=self.electricity,
            weather_df=weather_df,
            dt_col=dt_col,
            temp_col=temp_col,
        )
        if not copy:
            self.electricity = normalized
        return normalized

    def per_capita_consumption(
        self,
        population_col: str = "Population and dwelling counts / Population, 2021",
    ) -> pd.Series:
        if self.electricity is None:
            raise ValueError(f"FSA '{self.code}' has no electricity series.")
        return compute_per_capita_series(
            load_series=self.electricity,
            census_row=self.census,
            population_col=population_col,
        )

    def apply_prism(
        self,
        weather_df: pd.DataFrame,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        models: tuple[str, ...] = ("2ch", "2cl", "3seg"),
    ) -> dict:
        if self.electricity is None:
            raise ValueError(f"FSA '{self.code}' has no electricity series.")
        return fit_prism_segmented(
            load_series=self.electricity,
            weather_df=weather_df,
            dt_col=dt_col,
            temp_col=temp_col,
            models=models,
        )

    def apply_heating_prism(
        self,
        weather_df: pd.DataFrame,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        base_temp_candidates: list[float] | None = None,
    ) -> dict:
        if self.electricity is None:
            raise ValueError(f"FSA '{self.code}' has no electricity series.")
        return prism_heating_change_point_summary(
            load_series=self.electricity,
            weather_df=weather_df,
            dt_col=dt_col,
            temp_col=temp_col,
            base_temp_candidates=base_temp_candidates,
        )

    def short_term_metrics(
        self,
        am_hours: tuple[int, int] = (6, 11),
        pm_hours: tuple[int, int] = (16, 21),
        dtw_labeler=None,
        use_dtw_clustering: bool = False,
        dtw_k_min: int = 2,
        dtw_k_max: int = 6,
        dtw_min_days: int = 20,
        dtw_dominance_threshold: float = 0.6,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if self.electricity is None:
            raise ValueError(f"FSA '{self.code}' has no electricity series.")
        dtw_labels = None
        if use_dtw_clustering:
            cluster_result = cluster_daily_profiles_dtw(
                load_series=self.electricity,
                k_min=dtw_k_min,
                k_max=dtw_k_max,
                min_days=dtw_min_days,
                dominance_threshold=dtw_dominance_threshold,
                show_progress=show_progress,
            )
            dtw_labels = cluster_result["daily_labels"]
        return compute_daily_short_term_metrics(
            load_series=self.electricity,
            am_hours=am_hours,
            pm_hours=pm_hours,
            dtw_labeler=dtw_labeler,
            dtw_labels=dtw_labels,
            show_progress=show_progress,
        )

    def plot_prism(
        self,
        weather_df: pd.DataFrame,
        dt_col: str = "date_time_local",
        temp_col: str = "temperature",
        models: tuple[str, ...] = ("2ch", "2cl", "3seg"),
        sample_size: int | None = 10000,
        random_state: int = 42,
        point_alpha: float = 0.35,
        point_size: int = 5,
        show_confidence_band: bool = True,
        height: int = 650,
        title: str | None = None,
        show: bool = True,
    ):
        return plot_fsa_prism_fit(
            fsa=self,
            weather_df=weather_df,
            dt_col=dt_col,
            temp_col=temp_col,
            models=models,
            sample_size=sample_size,
            random_state=random_state,
            point_alpha=point_alpha,
            point_size=point_size,
            show_confidence_band=show_confidence_band,
            height=height,
            title=title,
            show=show,
        )
