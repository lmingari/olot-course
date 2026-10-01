import functools

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as crs
import cartopy.feature as cfeature

def refresh_if_needed(method):
    """Re-display the figure if the map already existed before this call.

    Wrap any plotter method that mutates self.ax with this. If self.ax was
    None (first call, map not created yet), the normal Jupyter inline-backend
    auto-display handles rendering. If self.ax already existed (a later-cell
    modification of a previously shown figure), this forces a re-display.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        map_existed = self.ax is not None
        result = method(self, *args, **kwargs)
        if map_existed:
            self._refresh()
        return result
    return wrapper

class MapPlotter:

    def __init__(self, extent=None):
        self.extent = extent
        self.fig = None
        self.ax = None

    def clean(self):
        """Close the current figure and reset the plot state."""
        if self.fig is not None:
            plt.close(self.fig)    # harmless if already closed
        self.fig = None
        self.ax = None
        return self
    
    def map(self):
        self.fig, self.ax = plt.subplots(
            subplot_kw={"projection": self._get_projection()}
        )

        if self.extent is not None:
            self.ax.set_extent(self.extent, crs=crs.PlateCarree())

        self.ax.add_feature(
            cfeature.LAND,
            facecolor="lightgrey",
            alpha=0.8,
        )
        self.ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.4,
        )

        gridlines = self.ax.gridlines(
            draw_labels=["left", "bottom"],
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gridlines.ylabel_style = {"rotation": 89}

        return self

    def add_marker(self, lon, lat, **kwargs):
        if self.ax is None:
            self.map()

        options = {
            "marker": "^",
            "color": "red",
            "s": 20,
            "transform": crs.PlateCarree(),
            "zorder": 10,
            "alpha": 0.5,
        } | kwargs

        self.ax.scatter(lon, lat, **options)

        return self

    def add_legend(self, **kwargs):
        if self.ax is None:
            raise RuntimeError("No plot created. Call a plotting method first.")
    
        self.ax.legend(**kwargs)
    
        return self
    
    def save(self, filename, **kwargs):
        if self.fig is None:
            raise RuntimeError("No map has been created yet.")

        self.fig.savefig(filename, bbox_inches="tight", **kwargs)

    def _crosses_antimeridian(self):
        if self.extent is None:
            return False
        x1, x2 = self.extent[:2]
        return x1 < 180 < x2 or x1 < -180 < x2

    def _get_projection(self):
        if self._crosses_antimeridian():
            return crs.PlateCarree(central_longitude=180)
        return crs.PlateCarree()

    def _refresh(self):
        """Re-display the figure, needed when modifying an ax created in a previous cell."""
        display(self.fig)

class DataPlotter(MapPlotter):
    contourf_defaults = {
        "cmap": plt.cm.RdYlBu_r,
        "extend": "max",
        "cbar_kwargs": {"shrink": 0.6},
    }
    pcolor_defaults = {
        "cmap": plt.cm.RdYlBu_r,
        "shading": "auto",
    }
    
    def __init__(self, data=None):
        super().__init__()
        self.data = None
        self.cbar = None

        if data is not None:
            self.set_data(data)

    def clean(self):
        super().clean()
        self.cbar = None
        return self

    def set_data(self, data):
        self.data = data

        x1, x2 = data.lon.min().item(), data.lon.max().item()
        y1, y2 = data.lat.min().item(), data.lat.max().item()
        self.extent = [x1, x2, y1, y2]

        self.clean()
        return self

    @refresh_if_needed
    def contourf(self, **kwargs):
        if self.data is None:
            raise RuntimeError("No data set. Call set_data() before contourf().")

        if self.ax is None:
            self.map()

        options = self.contourf_defaults | kwargs

        self.data.plot.contourf(
            ax=self.ax,
            transform=crs.PlateCarree(),
            **options,
        )
        return self

    @refresh_if_needed
    def contour(self, **kwargs):
        if self.data is None:
            raise RuntimeError("No data set. Call set_data() before contour().")

        if self.ax is None:
            self.map()

        cs = self.data.plot.contour(
            ax=self.ax,
            transform=crs.PlateCarree(),
            **kwargs,
        )
        self.ax.clabel(cs, inline=True, fontsize=9)
        return self

    @refresh_if_needed
    def pcolormesh(self, **kwargs):
        if self.data is None:
            raise RuntimeError(
                "No data set. Call set_data() before pcolormesh()."
            )
    
        if self.ax is None:
            self.map()

        options = self.pcolor_defaults | kwargs
    
        mesh = self.ax.pcolormesh(
            self.data.lon,
            self.data.lat,
            self.data,
            transform=crs.PlateCarree(),
            **options,
        )

        self.cbar = self.fig.colorbar(
            mesh,
            orientation="horizontal",
            shrink=0.4
        )
    
        self._post_pcolor()
        return self

    def _post_pcolor(self):
        """Hook: called after pcolormesh draws. Override in subclasses."""

class DecisionBoundariesPlotter(DataPlotter):
    pcolor_defaults = {
        "cmap": ListedColormap(["lightgreen", "moccasin", "lightcoral"]),
        "shading": "auto",
        "alpha": 0.5,
    }
    def _post_pcolor(self):
        if self.cbar is not None:
            self.cbar.set_ticks([0, 1, 2])
            self.cbar.set_ticklabels(["Low", "Moderate", "High"])
