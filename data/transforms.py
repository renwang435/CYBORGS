import random
from typing import Tuple

import albumentations as alb
import cv2
from PIL import Image, ImageFilter, ImageOps

cv2.setNumThreads(1)
import ipdb
import numpy as np
import torch


class ColorJitter(alb.ImageOnlyTransform):
    r"""
    Randomly change brightness, contrast, hue and saturation of the image. This
    class behaves exactly like :class:`torchvision.transforms.ColorJitter` but
    is slightly faster (uses OpenCV) and compatible with rest of the transforms
    used here (albumentations-style). This class works only on ``uint8`` images.

    .. note::

        Unlike torchvision variant, this class follows "garbage-in, garbage-out"
        policy and does not check limits for jitter factors. User must ensure
        that ``brightness``, ``contrast``, ``saturation`` should be ``float``
        in ``[0, 1]`` and ``hue`` should be a ``float`` in ``[0, 0.5]``.

    Parameters
    ----------
    brightness: float, optional (default = 0.4)
        How much to jitter brightness. ``brightness_factor`` is chosen
        uniformly from ``[1 - brightness, 1 + brightness]``.
    contrast: float, optional (default = 0.4)
        How much to jitter contrast. ``contrast_factor`` is chosen uniformly
        from ``[1 - contrast, 1 + contrast]``
    saturation: float, optional (default = 0.4)
        How much to jitter saturation. ``saturation_factor`` is chosen
        uniformly from ``[1 - saturation, 1 + saturation]``.
    hue: float, optional (default = 0.4)
        How much to jitter hue. ``hue_factor`` is chosen uniformly from
        ``[-hue, hue]``.
    always_apply: bool, optional (default = False)
        Indicates whether this transformation should be always applied.
    p: float, optional (default = 0.5)
        Probability of applying the transform.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, img, **params):
        original_dtype = img.dtype

        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        # Convert arguments as required by albumentations functional interface.
        # "gain" = contrast and "bias" = (brightness_factor - 1)
        img = alb.augmentations.functional.brightness_contrast_adjust(
            img, alpha=contrast_factor, beta=brightness_factor - 1
        )
        # Hue and saturation limits are required to be integers.
        img = alb.augmentations.functional.shift_hsv(
            img,
            hue_shift=int(hue_factor * 255),
            sat_shift=int(saturation_factor * 255),
            val_shift=0,
        )
        img = img.astype(original_dtype)
        return img

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "saturation", "hue")


class ToTensorV2(alb.BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


# =============================================================================

IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)
r"""ImageNet color normalization mean in RGB format (values in 0-1)."""

IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)
r"""ImageNet color normalization std in RGB format (values in 0-1)."""

# =============================================================================

