"""Tests for ClimateVision ML models."""

import pytest
import torch

from climatevision.models.unet import UNet
from climatevision.models.siamese import SiameseNetwork


@pytest.mark.parametrize(
    "n_channels,n_classes",
    [
        (4, 2),  # deforestation
        (4, 3),  # ice_melting
        (3, 3),  # flooding
    ],
)
def test_unet_init(n_channels: int, n_classes: int) -> None:
    """U-Net should initialize with variable input/output shapes."""
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    assert model.n_channels == n_channels
    assert model.n_classes == n_classes


def test_unet_forward_shape() -> None:
    """U-Net forward should preserve spatial dimensions."""
    model = UNet(n_channels=4, n_classes=2)
    x = torch.randn(1, 4, 256, 256)
    logits = model(x)
    assert logits.shape == (1, 2, 256, 256)


def test_siamese_forward_shape() -> None:
    """Siamese network should output a change map."""
    model = SiameseNetwork(in_channels=4)
    before = torch.randn(1, 4, 256, 256)
    after = torch.randn(1, 4, 256, 256)
    logits = model(before, after)
    assert logits.shape == (1, 2, 256, 256)
