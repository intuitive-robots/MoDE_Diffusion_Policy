from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
import einops

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0



def set_parameter_requires_grad(model, requires_grad):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad
            
            
def freeze_params(model):
    set_parameter_requires_grad(model, requires_grad=False)


class FilmModule(nn.Module):
    """
    FilmModule modulation for conditioning.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_size, 4 * hidden_size, bias=True)
        ).to(dtype=torch.get_default_dtype())
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)
        
    def forward(self, c):
        x = self.modulation(c).chunk(2, dim=-1)
        
        return x[0].chunk(2, dim=-1), x[1].chunk(2, dim=-1)


class BasicBlockWithModulation(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, modulation_params=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if modulation_params is not None:
            # Reshape gamma and beta for broadcasting
            gamma, beta = modulation_params
            gamma = gamma.view(-1, self.bn2.num_features, 1, 1)
            beta = beta.view(-1, self.bn2.num_features, 1, 1)

            # Apply FiLM modulation after the second batch normalization
            out = (gamma) * out  + beta

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # print(out.shape)
        return out
    

class ResNetEncoderWithFiLM(nn.Module):
    def __init__(self, cond_dim, latent_dim=128, pretrained=False, hidden_size=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.inplanes = 64  # Initialize inplanes attribute
        backbone = models.resnet18(pretrained=pretrained)
        n_inputs = backbone.fc.in_features
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Instantiate the FilmModule
        # print(f'cond vector dim: {cond_dim}')
        self.film_module1 = FilmModule(input_size=cond_dim, hidden_size=64)
        self.film_module2 = FilmModule(input_size=cond_dim, hidden_size=128)
        self.film_module3 = FilmModule(input_size=cond_dim, hidden_size=256)
        self.film_module4 = FilmModule(input_size=cond_dim, hidden_size=512)

        # Replace the original ResNet layers with modulation-enabled layers
        self.layer1 = self._make_modulated_layer(backbone.layer1, BasicBlockWithModulation)
        self.inplanes = 64  # Update inplanes for layer2
        self.layer2 = self._make_modulated_layer(backbone.layer2, BasicBlockWithModulation)
        self.inplanes = 128  # Update inplanes for layer3
        self.layer3 = self._make_modulated_layer(backbone.layer3, BasicBlockWithModulation)
        self.inplanes = 256  # Update inplanes for layer4
        self.layer4 = self._make_modulated_layer(backbone.layer4, BasicBlockWithModulation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_inputs, latent_dim)

    def _make_modulated_layer(self, layer, block):
        layers = []
        for blk in layer:
            downsample = None
            if blk.downsample is not None or self.inplanes != blk.conv1.out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, blk.conv1.out_channels, 1, stride=blk.stride, bias=False),
                    nn.BatchNorm2d(blk.conv1.out_channels),
                )
            layers.append(block(self.inplanes, blk.conv1.out_channels, blk.stride, downsample))
            self.inplanes = blk.conv1.out_channels
        return nn.Sequential(*layers)

    def forward(self, x, conditioning_vector=None):
        # print(f'Beginnign cond vector shape {conditioning_vector.shape}' )
        batch_size = len(x)
        t_steps = 1
        time_series = False
        
        if len(x.shape) == 5:
            # print('reshaping')
            t_steps = x.shape[1]
            x = einops.rearrange(x, 'b t n x_dim y_dim -> (b t) n x_dim y_dim')
            # print(f'After rearrange x shape: {x.shape}')
            time_series = True
            
            repeat_factor = t_steps  # How many times you want to repeat
            conditioning_vector = torch.cat([conditioning_vector for _ in range(repeat_factor)], dim=0)
            # print(conditioning_vector.shape)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        if conditioning_vector is not None:
            # Generate modulation parameters from the conditioning vector
            # print(f'cond vector shape {conditioning_vector.shape}' )
            # print(self.film_module1)
            modulation_params1 = self.film_module1(conditioning_vector)
            modulation_params2 = self.film_module2(conditioning_vector)
            modulation_params3 = self.film_module3(conditioning_vector)
            modulation_params4 = self.film_module4(conditioning_vector)
            
            # print(modulation_params1[0][0].shape)
            # print(x.shape)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Manually forward through each block with modulation
            for idx, block in enumerate(self.layer1):
                x = block(x, modulation_params1[idx])
            for idx, block in enumerate(self.layer2):
                x = block(x, modulation_params2[idx])
            for idx, block in enumerate(self.layer3):
                x = block(x, modulation_params3[idx])
            for idx, block in enumerate(self.layer4):
                x = block(x, modulation_params4[idx])
        else:
            # Standard ResNet forward pass without modulation
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        # print(f'After model process x shape: {x.shape}')
        x = self.avgpool(x)
        # print(f'After avgpool x shape: {x.shape}')
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        # print('done')
        if time_series:
            x = einops.rearrange(x, '(b t) d -> b t d', b=batch_size, t=t_steps, d=self.latent_dim)       
        return x



class ResNetTokenEncoderWithFiLM(nn.Module):
    def __init__(self, cond_dim, latent_dim=128, pretrained=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.inplanes = 64  # Initialize inplanes attribute
        backbone = models.resnet18(pretrained=pretrained)
        n_inputs = backbone.fc.in_features
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Instantiate the FilmModule
        self.film_module1 = FilmModule(input_size=cond_dim, hidden_size=64)
        self.film_module2 = FilmModule(input_size=cond_dim, hidden_size=128)
        self.film_module3 = FilmModule(input_size=cond_dim, hidden_size=256)
        self.film_module4 = FilmModule(input_size=cond_dim, hidden_size=512)

        # Replace the original ResNet layers with modulation-enabled layers
        self.layer1 = self._make_modulated_layer(backbone.layer1, BasicBlockWithModulation)
        self.inplanes = 64  # Update inplanes for layer2
        self.layer2 = self._make_modulated_layer(backbone.layer2, BasicBlockWithModulation)
        self.inplanes = 128  # Update inplanes for layer3
        self.layer3 = self._make_modulated_layer(backbone.layer3, BasicBlockWithModulation)
        self.inplanes = 256  # Update inplanes for layer4
        self.layer4 = self._make_modulated_layer(backbone.layer4, BasicBlockWithModulation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_inputs, latent_dim)

    def _make_modulated_layer(self, layer, block):
        layers = []
        for blk in layer:
            downsample = None
            if blk.downsample is not None or self.inplanes != blk.conv1.out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, blk.conv1.out_channels, 1, stride=blk.stride, bias=False),
                    nn.BatchNorm2d(blk.conv1.out_channels),
                )
            layers.append(block(self.inplanes, blk.conv1.out_channels, blk.stride, downsample))
            self.inplanes = blk.conv1.out_channels
        return nn.Sequential(*layers)

    def forward(self, x, conditioning_vector=None):
        # print(f'Beginnign cond vector shape {conditioning_vector.shape}' )
        batch_size = len(x)
        t_steps = 1
        time_series = False
        
        if len(x.shape) == 5:
            # print('reshaping')
            t_steps = x.shape[1]
            x = einops.rearrange(x, 'b t n x_dim y_dim -> (b t) n x_dim y_dim')
            # print(f'After rearrange x shape: {x.shape}')
            time_series = True
            
            repeat_factor = t_steps  # How many times you want to repeat
            conditioning_vector = torch.cat([conditioning_vector for _ in range(repeat_factor)], dim=0)
            # print(conditioning_vector.shape)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        if conditioning_vector is not None:
            # Generate modulation parameters from the conditioning vector
            # print(f'cond vector shape {conditioning_vector.shape}' )
            modulation_params1 = self.film_module1(conditioning_vector)
            modulation_params2 = self.film_module2(conditioning_vector)
            modulation_params3 = self.film_module3(conditioning_vector)
            modulation_params4 = self.film_module4(conditioning_vector)
            
            # print(modulation_params1[0][0].shape)
            # print(x.shape)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Manually forward through each block with modulation
            for idx, block in enumerate(self.layer1):
                x = block(x, modulation_params1[idx])
            for idx, block in enumerate(self.layer2):
                x = block(x, modulation_params2[idx])
            for idx, block in enumerate(self.layer3):
                x = block(x, modulation_params3[idx])
            for idx, block in enumerate(self.layer4):
                x = block(x, modulation_params4[idx])
        else:
            # Standard ResNet forward pass without modulation
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        if time_series:
            x = einops.rearrange(x, '(b t) d -> b t d', b=batch_size, t=t_steps, d=self.latent_dim)       
        return x