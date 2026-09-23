"""
Augmentation pipelines for SkinFLNet++.

Training: RandAugment + flips + ColorJitter + Normalize
Val/Test: Resize + CenterCrop + Normalize
"""

import torchvision.transforms as T


# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> T.Compose:
    """Training augmentation pipeline as specified in §9.5 of the spec."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> T.Compose:
    """Validation / test transform: Resize → CenterCrop → Normalize."""
    return T.Compose([
        T.Resize(int(img_size * 1.14)),  # ~256 for 224
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
