from __future__ import annotations

from typing import Sequence

import torch
from torchvision.transforms import v2

# CLIP normalization constants (OpenAI CLIP preprocessing).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Default normalization for DINO-style image pipelines.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _gaussian_blur(
    p: float,
    kernel_size: int = 9,
    sigma: tuple[float, float] = (0.1, 2.0),
) -> v2.RandomApply:
    return v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p=p)


class Stage1DINOMultiCropTransform:
    """DINO-style multi-crop augmentation for Stage1."""

    def __init__(
        self,
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        global_crops_number: int = 2,
        local_crops_number: int = 8,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=global_crop_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=local_crop_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        color_jitter = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )
        normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        self.global_transform_1 = v2.Compose(
            [
                color_jitter,
                _gaussian_blur(p=1.0),
                normalize,
            ]
        )
        self.global_transform_2 = v2.Compose(
            [
                color_jitter,
                _gaussian_blur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
                normalize,
            ]
        )
        self.local_transform = v2.Compose(
            [
                color_jitter,
                _gaussian_blur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        global_crops = []
        for crop_idx in range(self.global_crops_number):
            base_crop = self.geometric_augmentation_global(image)
            if crop_idx == 0:
                global_crops.append(self.global_transform_1(base_crop))
            else:
                global_crops.append(self.global_transform_2(base_crop))

        local_crops = [
            self.local_transform(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]

        return {"global_crops": global_crops, "local_crops": local_crops}


def build_stage1_dino_multicrop_transform(
    global_crops_scale: tuple[float, float] = (0.4, 1.0),
    local_crops_scale: tuple[float, float] = (0.05, 0.4),
    global_crops_number: int = 2,
    local_crops_number: int = 8,
    global_crop_size: int = 224,
    local_crop_size: int = 96,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Stage1DINOMultiCropTransform:
    return Stage1DINOMultiCropTransform(
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        global_crops_number=global_crops_number,
        local_crops_number=local_crops_number,
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        mean=mean,
        std=std,
    )


def build_stage2_clip_transform(
    image_size: int = 224,
    mean: Sequence[float] = CLIP_MEAN,
    std: Sequence[float] = CLIP_STD,
) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(image_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

