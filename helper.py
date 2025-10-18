import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Dataset for deposit thickness
class ThicknessDataset(Dataset):
    def __init__(self, X, y, transform = None):
        """
        X: numpy array (N, 2) with [lat, lon]
        y: numpy array (N,)
        """
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

# Dataset for ensemble forecasts
class EnsembleDataset(Dataset):
    def __init__(self, data_array, transform = None):
        """
        Parameters
        ----------
        data_array : xr.DataArray
            Input 3D xarray DataArray with dimensions (ens, lat, lon), where:
            - ens: ensemble members (e.g., different model runs or simulations)
            - lat: latitude coordinates
            - lon: longitude coordinates
        """
        self.X = data_array.values
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        x = x.unsqueeze(0)  # add channel dimension
        return x
        
##############
### Models ###
##############
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        ### Encoder ###
        self.encoder = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3, stride=2, padding=1), # N,16,51,61
                nn.ReLU(True),
                nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1), #N,32,26,31
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #N,64,13,16
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(64*13*16,latent_dim)
                )

        ### Decoder ###
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64*13*16),
                nn.ReLU(True),
                nn.Unflatten(1, (64,13,16)),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                nn.ReLU(True)
                )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self,x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        ### Encoder ###
        self.encoder = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3, stride=2, padding=1), # N,16,51,61
                nn.ReLU(True),
                nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1), #N,32,26,31
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #N,64,13,16
                nn.ReLU(True),
                nn.Flatten(),
                )

        self.fc_mu     = nn.Linear(64*13*16,latent_dim)
        self.fc_logvar = nn.Linear(64*13*16,latent_dim)

        ### Decoder ###
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64*13*16),
                nn.ReLU(True),
                nn.Unflatten(1, (64,13,16)),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                nn.ReLU(True)
                )
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

    def encode(self,x):
        h = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

##################
### Criterions ###
##################
class VAELoss(nn.Module):
    def __init__(self, beta = 1.0):
        """
        reduction: 'sum' or 'mean'
        beta: weight on the KL divergence term (beta-VAE variant)
        """
        super().__init__()
        self.beta = beta

    def forward(self,prediction,x,mu,logvar):
        # Reconstruction loss
        loss1 = F.mse_loss(prediction, x, reduction="sum")

        # KL divergence
        loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = loss1 + self.beta * loss2

        return loss / x.size(0)
