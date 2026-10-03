import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

#######################
### Transformations ###
#######################
class Standardize:
    """y = (x - mean) / (std + eps).

    x: CPU tensor (B, 1, H, W).
    mean, std: scalars or arrays that broadcast against x, e.g. (H, W),
    computed on the training data.
    """

    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.eps = eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def invert(self, x):
        return x * (self.std + self.eps) + self.mean


class MinMaxScale:
    """y = (x - min) / (max - min), so the training data lies in [0, 1].

    x: CPU tensor (B, 1, H, W).
    min_value, max_value: scalars or arrays that broadcast against x.
    """

    def __init__(self, min_value, max_value):
        self.min = torch.as_tensor(min_value, dtype=torch.float32)
        self.max = torch.as_tensor(max_value, dtype=torch.float32)

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

    def invert(self, x):
        return x * (self.max - self.min) + self.min

class LogTransform:
    """Compress the dynamic range of a non-negative field and rescale it.

        forward: y = log(1 + factor * x) / scale
        invert:  x = (exp(y * scale) - 1) / factor

    Works on torch tensors (CPU or GPU), so ``invert`` can be applied directly
    to samples generated on the GPU.

    Notes
    -----
    * x must satisfy x > -1 / factor (in practice: x >= 0), otherwise log1p gives NaN.
    * Flow matching starts from N(0, 1) noise, so ``scale`` should make the
      transformed data roughly unit-variance. A simple choice is the standard
      deviation of ``log1p(x * factor)`` over the training data.
    """

    def __init__(self, factor=1.0, scale=1.0):
        self.factor = factor
        self.scale = scale

    def __call__(self, x):
        return torch.log1p(x * self.factor) / self.scale

    def invert(self, x):
        return torch.expm1(x * self.scale) / self.factor

################
### Datasets ###
################

class SupervisedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        """
        X: NumPy array (N, features)
        y: NumPy array (N,), integer class labels
        transform: optional callable on a CPU tensor (N, features), applied once here
        """
        x = torch.as_tensor(X, dtype=torch.float32)        # (N, features)
        if transform is not None:
            x = transform(x)

        self.x = x
        self.y = torch.as_tensor(y, dtype=torch.long)      # (N,)
        self.transform = transform                         # kept to invert later

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]                    # (features,), scalar


class SuperResolutionDataset(Dataset):
    def __init__(self, da: xr.DataArray,
                 scale_factor: tuple[int, int] = (4, 4),
                 transform=None):
        """
        da: xarray.DataArray (ens, lat, lon), the high-resolution fields
        scale_factor: (lat_factor, lon_factor), coarsening of the low-res input
        transform: optional callable on a CPU tensor (N, 1, H, W), applied once
                   here to both the low-res and the high-res fields
        """
        lat_factor, lon_factor = scale_factor

        hr = da.values                                                       # (N, H, W)
        lr = da.coarsen(lat=lat_factor, lon=lon_factor).mean().values        # (N, H/fy, W/fx)

        hr = torch.as_tensor(hr, dtype=torch.float32)[:, None]               # (N, 1, H, W)
        lr = torch.as_tensor(lr, dtype=torch.float32)[:, None]               # (N, 1, H/fy, W/fx)

        if transform is not None:
            hr = transform(hr)
            lr = transform(lr)

        self.hr = hr
        self.lr = lr
        self.transform = transform                                           # kept to invert later

    def __len__(self):
        return self.hr.shape[0]

    def __getitem__(self, index):
        return self.lr[index], self.hr[index]                                # (1, H/fy, W/fx), (1, H, W)

class SimulationDataset(Dataset):
    """2D simulation fields stored in an xarray DataArray.

    Args:
        da:        xarray.DataArray with dimensions (ens, lat, lon).
        transform: optional callable applied once to the whole array
                   (e.g. LogTransform). Keep it to invert samples later.
        coarsen:   optional (lat_factor, lon_factor), e.g. (2, 2). Averages
                   blocks of cells, so a 256x256 grid becomes 128x128.
                   lat and lon sizes must be divisible by the factors.
                   Coarsening is applied BEFORE the transform.

    Each item is a float32 tensor of shape (1, lat, lon): one ensemble member
    with a single channel. A DataLoader therefore yields (B, 1, lat, lon).
    """

    def __init__(self, da, transform=None, coarsen=None):

        # Optional block-average coarsening: (ens, lat, lon) -> (ens, lat/fy, lon/fx).
        if coarsen is not None:
            fy, fx = coarsen
            da = da.coarsen(lat=fy, lon=fx).mean()

        data = torch.as_tensor(da.values, dtype=torch.float32)   # (N, H, W)
        data = data[:, None]                                     # (N, 1, H, W)

        # Transform once here instead of at every __getitem__ call.
        if transform is not None:
            data = transform(data)

        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]                                  # (1, H, W)


###############
### helpers ###
###############
def train_validation_split(X, y, train_fraction=0.8, seed=42):
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))
    n_train = round(train_fraction * len(X))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return (
        X[train_idx],
        X[val_idx],
        y[train_idx],
        y[val_idx],
    )
