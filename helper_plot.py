import numpy as np
import matplotlib
import matplotlib.pyplot as plt                  # plots and visualizations
import cartopy.crs as crs                        # coordinate systems for maps
import cartopy.feature as cfeature

import torch

def create_map(projection = crs.PlateCarree()):
    fig, ax = plt.subplots( subplot_kw={'projection': projection} )
    ###
    ### Add map features
    ###
    BORDERS = cfeature.NaturalEarthFeature(
            scale     = '10m',
            category  = 'cultural',
            name      = 'admin_0_countries',
            edgecolor = 'gray',
            facecolor = 'none'
            )
    LAND = cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            edgecolor = 'none',
            facecolor = 'lightgrey',
            alpha     = 0.8
            )
    
    ax.add_feature(LAND,zorder=0)
    ax.add_feature(BORDERS, linewidth=0.4)
    ###
    ### Add grid lines
    ###
    gl = ax.gridlines(
        crs         = crs.PlateCarree(),
        draw_labels = True,
        linewidth   = 0.5,
        color       = 'gray',
        alpha       = 0.5,
        linestyle   = '--')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.ylabel_style  = {'rotation': 90}

    fig.tight_layout()
    
    return fig, ax

def plot_decision_regions(model, transform):
    # Define the grid resolution
    lat_min, lat_max = 28.4, 28.9
    lon_min, lon_max = -18.2, -17.65
    n_lat, n_lon = 220, 220   # resolution of the grid

    # Create coordinate arrays
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    # Meshgrid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")  # shape (n_lat, n_lon)

    # Flatten to feed into model
    coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
    coords_torch = torch.tensor(coords, dtype=torch.float32)

    X = transform(coords_torch)

    # Predict
    with torch.no_grad():
        logits = model(X)
        prediction = torch.argmax(logits, dim=1).numpy()

    fig, ax = create_map()

    cb = ax.contourf(lons, lats,
        prediction.reshape(n_lat, n_lon),
        vmin=0,
        vmax=2,
        levels = [-0.5, 0.5, 1.5, 2.5],
        alpha=0.5,
        )

    cbar = fig.colorbar(cb,
        orientation = 'horizontal',
        shrink = 0.4,
        #aspect = 40,
    )
    cbar.set_ticks([0,1,2])
    cbar.set_ticklabels(['Low', 'Moderate', 'High'])

    return fig, ax
