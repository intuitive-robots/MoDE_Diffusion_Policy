from multiprocessing.sharedctypes import Value

import hydra
from torch import DictType, nn
from .utils import append_dims
import torch
import numpy as np

'''
Wrappers for the score-based models based on Karras et al. 2022
They are used to get improved scaling of different noise levels, which
improves training stability and model performance 

Code is adapted from:

https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
'''

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)
    



class GCDenoiser(nn.Module):
    def __init__(self, inner_model, sigma_data=1., logvar_channels=128, **unet_kwargs):
        super().__init__()
        self.inner_model = hydra.utils.instantiate(inner_model)
        self.sigma_data = sigma_data
        
        # Add the uncertainty prediction layers (similar to Precond).
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state, action, goal, noise, sigma, return_logvar=True, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)

        # Incorporate log variance prediction, if requested.
        model_output, logvar = self.forward(state, noised_input, goal, sigma, return_logvar=return_logvar, **kwargs)
        
        # Compute the target as before.
        target = (action - c_skip * noised_input) / c_out
        
        # Modify loss to match EDM2Loss using the uncertainty weighting.
        if return_logvar:
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            loss = (weight / logvar.exp()) * ((model_output - target) ** 2) + logvar
        else:
            loss = (model_output - target).pow(2).flatten(1).mean()
        
        return loss.mean(), model_output

    def forward(self, state, action, goal, sigma, return_logvar=False, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        x_in = (action * c_in)
        # Pass the modified input through the inner model.
        model_output = self.inner_model(state, x_in, goal, sigma, **kwargs)
        
        # Compute D(x), similar to Precond.
        D_x = c_skip * action + c_out * model_output
        
        if return_logvar:
            # Predict log variance using the logvar_fourier and logvar_linear layers.
            logvar = self.logvar_linear(self.logvar_fourier(sigma)).reshape(-1, 1, 1, 1)
            return D_x, logvar
        
        return D_x

