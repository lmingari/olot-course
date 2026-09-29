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

    def map(self):
        self.fig, self.ax = plt.subplots(
            subplot_kw={"projection": crs.PlateCarree()}
        )

        if self.extent is not None:
            self.ax.set_extent(self.extent)

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

    def _refresh(self):
        """Re-display the figure, needed when modifying an ax created in a previous cell."""
        display(self.fig)

class DataPlotter(MapPlotter):

    def __init__(self, data=None):
        super().__init__()
        self.data = None

        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        self.data = data

        x1, x2 = data.lon.min(), data.lon.max()
        y1, y2 = data.lat.min(), data.lat.max()
        self.extent = [x1, x2, y1, y2]

        # New data means a fresh start
        self.fig = None
        self.ax = None

        return self

    @refresh_if_needed
    def contourf(self, **kwargs):
        if self.data is None:
            raise RuntimeError("No data set. Call set_data() before contourf().")

        if self.ax is None:
            self.map()

        options = {
            "cmap": plt.cm.RdYlBu_r,
            "extend": "max",
        } | kwargs

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

        cmap = ListedColormap(["lightgreen", "moccasin", "lightcoral"])
    
        options = {
            "cmap": cmap,
            "shading": "auto",
            "alpha": 0.5,
        } | kwargs
    
        mesh = self.ax.pcolormesh(
            self.data.lon,
            self.data.lat,
            self.data,
            transform=crs.PlateCarree(),
            **options,
        )

        cbar = self.fig.colorbar(
            mesh,
            orientation="horizontal",
            shrink=0.4,
        )
    
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Low", "Moderate", "High"])
    
        return self