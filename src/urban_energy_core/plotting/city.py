from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from urban_energy_core.plotting._legacy import plot_city_fsa_map


def plot_spatial_samples_with_basemap(
    *,
    fsa_gdf: Any,
    da_gdf: Any,
    building_gdf: Any,
    figsize: tuple[float, float] = (20, 6),
    zoom: int = 11,
    titles: tuple[str, str, str] = (
        "Sampled Montreal FSAs",
        "Sampled Montreal DAs",
        "Sampled Montreal Buildings",
    ),
):
    try:
        import contextily as ctx
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise ImportError(
            "contextily is required for OSM-backed sample maps. Install it with "
            "`pip install contextily` or `pip install -e .[notebooks]` after adding the extra."
        ) from exc

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    fsa_web = fsa_gdf.to_crs(epsg=3857)
    da_web = da_gdf.to_crs(epsg=3857)
    building_web = building_gdf.to_crs(epsg=3857)

    layers = [
        (fsa_web, axes[0], titles[0], "fsa"),
        (da_web, axes[1], titles[1], "da"),
        (building_web, axes[2], titles[2], "building"),
    ]

    for gdf, ax, title, layer_type in layers:
        if layer_type == "fsa":
            gdf.plot(
                ax=ax,
                edgecolor="#1f2937",
                facecolor="#93c5fd",
                linewidth=1.0,
                alpha=0.45,
            )
        elif layer_type == "da":
            gdf.plot(
                ax=ax,
                edgecolor="#92400e",
                facecolor="#fbbf24",
                linewidth=0.2,
                alpha=0.45,
            )
        else:
            gdf.plot(
                ax=ax,
                edgecolor="#1d4ed8",
                facecolor="none",
                linewidth=0.2,
                alpha=0.8,
            )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)

        ax.set_title(title, fontsize=13)
        ax.set_axis_off()

    plt.tight_layout()
    return fig, axes


__all__ = ["plot_city_fsa_map", "plot_spatial_samples_with_basemap"]
