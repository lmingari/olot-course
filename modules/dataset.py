import numpy as np
import torch
from torch.utils.data import Dataset

#######################
### Transformations ###
#######################
class Standardize:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.from_numpy(mean).float()
        self.std  = torch.from_numpy(std).float()
        self.eps  = eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def invert(self, x):
        return x * (self.std + self.eps) + self.mean

class MinMaxScale:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

    def invert(self, x):
        return x * (self.max - self.min) + self.min

class LogTransform:
    def __init__(self, factor=1000.0, scale=1.0):
        self.factor = factor
        self.scale = scale

    def __call__(self, x):
        return np.log1p(x * self.factor) / self.scale

    def invert(self, x):
        return np.expm1(x * self.scale) / self.factor

################
### Datasets ###
################

class SupervisedDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        X: NumPy array (N, features)
        y: NumPy array (N,)
        """
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, self.y[idx]

class SuperResolutionDataset:
    def __init__(self, da, 
                 scale_factor=(4, 4),
                 transform=None):         
        lat_factor, lon_factor = scale_factor 
        self.hr = da.values
        self.lr = da.coarsen(
            lat=lat_factor,
            lon=lon_factor
        ).mean().values

        self.transform = transform

    def __len__(self):
        return self.hr.shape[0]

    def __getitem__(self, index):
        x = self.lr[index][None, ...]
        y = self.hr[index][None, ...]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

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