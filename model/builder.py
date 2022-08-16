import copy
import random
from functools import wraps
from typing import Dict, Sequence, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision import transforms as T


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        # x is (batch, num_masks, ft_dim)
        x = rearrange(x, 'b m c -> m b c')
        out = [self.net(t) for t in x]
        out = torch.stack(out, dim=0)
        out = rearrange(out, 'm b c -> b m c')

        return out

# Mask downsampling and sampling for DetCon
class MaskPooling(nn.Module):
    def __init__(
        self, num_classes: int, num_samples: int = 16, downsample: int = 32
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)

    def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Create binary masks and performs mask pooling
        Args:
            masks: (b, 1, h, w)
        Returns:
            masks: (b, num_classes, d)
        """
        if masks.ndim < 4:
            masks = masks.unsqueeze(dim=1)

        masks = masks == self.mask_ids[None, :, None, None].to(masks.device)    # (b, num_classes, h, w)
        masks = self.pool(masks.to(torch.float))                                # (b, num_classes, h / downsample, w / downsample)
        masks = rearrange(masks, "b c h w -> b c (h w)")
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes).to(masks.device)[masks]
        masks = rearrange(masks, "b d c -> b c d")
        return masks

    def sample_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Samples which binary masks to use in the loss.
        Args:
            masks: (b, num_classes, d)
        Returns:
            masks: (b, num_samples, d)
        """
        bs = masks.shape[0]
        mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
        sel_masks = mask_exists.to(torch.float) + 1e-11

        # Try to make masks as diverse as possible
        if self.num_samples > sel_masks.shape[1]:
            mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples,
                                            replacement=True)
        else:
            mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples)
        sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
        return sampled_masks, mask_ids

    def forward(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        binary_masks = self.pool_masks(masks)
        sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
        area = sampled_masks.sum(dim=-1, keepdim=True)
        sampled_masks = sampled_masks / torch.maximum(area, torch.tensor(1.0))
        return sampled_masks, sampled_mask_ids


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, 
                num_classes=81,
                downsample=32,
                num_samples=16,
                layers = [-2]):
        super().__init__()
        self.net = net
        self.layers = layers

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.mask_pool = MaskPooling(num_classes, num_samples, downsample)

        self.hidden = defaultdict(list)
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layers[0]) == str:
            modules = dict([*self.net.named_modules()])
            return [modules.get(layer, None) for layer in self.layers]
        elif type(self.layers[0]) == int:
            children = [*self.net.children()]
            return [children[layer] for layer in self.layers]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device].append(output)

    def _register_hook(self):
        layers = self._find_layer()
        assert layers is not None, f'hidden layer ({self.layer}) not found'
        for layer in layers:
            handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layers[0] == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        # Check every hidden layer
        for i, layer_out in enumerate(hidden):
            assert layer_out is not None, f'hidden layer {self.layers[i]} never emitted an output'

        # Last hidden layer will have smallest feature map --> downsample to this resolution
        # Then concatenate features
        _, _, _, ft_map_dim = hidden[-1].shape
        final_hidden = hidden[-1]
        for j in reversed(range(len(self.layers) - 1)):
            _, _, _, curr_ft_map_dim = hidden[j].shape
            downsample_factor = curr_ft_map_dim // ft_map_dim
            final_hidden = torch.cat([F.avg_pool2d(hidden[j],
                                                    kernel_size=downsample_factor,
                                                    stride=downsample_factor,),
                                      final_hidden
                                    ], dim=1)

        return final_hidden

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> Sequence[torch.Tensor]:
        m, mids = self.mask_pool(masks)
        raw_ft_map = self.get_representation(x)

        # Pool representations according to masks
        e = rearrange(raw_ft_map, "b c h w -> b (h w) c")
        e = m @ e

        # Get projection
        projector = self._get_projector(e)
        p = projector(e)
        return raw_ft_map, p, m, mids


# main class
class DetConB(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layers=[-2],
        projection_size=256,
        projection_hidden_size=4096,
        num_classes=81,
        downsample=32,
        num_samples=16,
        moving_average_decay=0.99,
        use_momentum=True
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size,
                                        num_classes=num_classes,
                                        downsample=downsample,
                                        num_samples=num_samples,
                                        layers=hidden_layers)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device),
            torch.randn(2, image_size, image_size, device=device),
            torch.randn(2, 3, image_size, image_size, device=device),
            torch.randn(2, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, crop1, mask1, crop2, mask2):
        # encode and project
        online_ft_map1, online_proj1, _, online_ids1 = self.online_encoder(crop1, mask1)
        online_ft_map2, online_proj2, _, online_ids2 = self.online_encoder(crop2, mask2)

        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_ft_map1, target_proj1, _, target_ids1 = target_encoder(crop1, mask1)
            target_ft_map2, target_proj2, _, target_ids2 = target_encoder(crop2, mask2)
            target_proj1 = target_proj1.detach()
            target_proj2 = target_proj2.detach()
            target_ft_map1 = target_ft_map1.detach()
            target_ft_map2 = target_ft_map2.detach()

        return (online_pred1, online_pred2, 
                target_proj1, target_proj2, 
                online_ids1, online_ids2,
                target_ids1, target_ids2,
                online_ft_map1, online_ft_map2,
                target_ft_map1, target_ft_map2)
