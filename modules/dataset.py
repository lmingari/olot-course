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