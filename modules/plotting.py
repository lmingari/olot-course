import functools

import matplotlib.pyplot as plt
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


class Plotter:

    def __init__(self, data=None):
        self.data = None
        self.extent = None
        self.fig = None
        self.ax = None
        self._markers = []  # list of (lon, lat, kwargs) registered but drawn lazily

        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        self.data = data
        x1, x2 = data.lon.min(), data.lon.max()
        y1, y2 = data.lat.min(), data.lat.max()
        self.extent = [x1, x2, y1, y2]

        # new data means a fresh start
        self.fig = None
        self.ax = None
        self._markers = []

        return self

    def _create_map(self, projection=crs.PlateCarree()):
        self.fig, self.ax = plt.subplots(
            subplot_kw={"projection": projection}
        )

        if self.extent is not None:
            self.ax.set_extent(self.extent)  # [x1,x2,y1,y2]

        borders = cfeature.NaturalEarthFeature(
            scale="10m",
            category="cultural",
            name="admin_0_countries",
            edgecolor="gray",
            facecolor="none",
        )

        land = cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "10m",
            edgecolor="none",
            facecolor="lightgrey",
            alpha=0.8,
        )

        self.ax.add_feature(land, zorder=0)
        self.ax.add_feature(borders, linewidth=0.4)

        gridlines = self.ax.gridlines(
            crs=crs.PlateCarree(),
            draw_labels=['left', 'bottom'],
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gridlines.ylabel_style = {"rotation": 89}

        # Draw any markers that were registered before the map existed
        self._draw_all_markers()

    def _refresh(self):
        """Re-display the figure, needed when modifying an ax created in a previous cell."""
        display(self.fig)

    @refresh_if_needed
    def contourf(self, **kwargs):
        if self.data is None:
            raise RuntimeError("No data set. Call set_data() before contourf().")

        if self.ax is None:
            self._create_map()

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
            self._create_map()

        cs = self.data.plot.contour(
            ax=self.ax,
            transform=crs.PlateCarree(),
            **kwargs,
        )
        self.ax.clabel(cs, inline=True, fontsize=9)
        return self

    @refresh_if_needed
    def add_marker(self, lon, lat, **kwargs):
        """Register a marker to be drawn. If a map already exists, draw it now too."""
        self._markers.append((lon, lat, kwargs))

        if self.ax is None:
            self._create_map()
        else:
            self._draw_marker(lon, lat, kwargs)

        return self

    def _draw_marker(self, lon, lat, kwargs):
        defaults = {
            "marker": "^",
            "color": "red",
            "markersize": 10,
            "transform": crs.PlateCarree(),
            "zorder": 10,
        }
        options = defaults | kwargs
        self.ax.plot(lon, lat, **options)

    def _draw_all_markers(self):
        for lon, lat, kwargs in self._markers:
            self._draw_marker(lon, lat, kwargs)

    def save(self, filename, **kwargs):
        if self.fig is None:
            raise RuntimeError("No plot has been created yet.")

        self.fig.savefig(filename, bbox_inches="tight", **kwargs)

class DecisionBoundariesPlotter(Plotter):

    @refresh_if_needed
    def contour(self, **kwargs):
        if self.data is None:
            raise RuntimeError("No data set. Call set_data() before contour().")

        if self.ax is None:
            self._create_map()

        cs = self.data.plot.contourf(
            ax=self.ax,
            transform=crs.PlateCarree(),
            levels=[-0.5, 0.5, 1.5, 2.5],
            vmin=0,
            vmax=2,
            alpha=0.5,
            **kwargs,
        )

        cbar = self.fig.colorbar(
            cs,
            orientation="horizontal",
            shrink=0.4,
        )
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Low", "Moderate", "High"])

        return self