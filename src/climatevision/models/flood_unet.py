"""
Production flood detection models.

Uses segmentation-models-pytorch (smp) with EfficientNet-B7 encoder,
pretrained on ImageNet. Two variants:
  - FloodUNet: 5-channel input (S2 B03/B08/B11 + S1 VV/VH)
  - FloodUNetS2Only: 3-channel input (S2 B03/B08/B11)

Both output 3 classes: dry_land, permanent_water, flooded.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _FirstConvAdapter(nn.Module):
    """
    Adapts a pretrained encoder's first conv layer to accept a different
    number of input channels by averaging pretrained weights across the
    extra channels.
    """

    def __init__(self, encoder: nn.Module, new_in_channels: int):
        super().__init__()
        self.encoder = encoder
        self.new_in_channels = new_in_channels

        # Find the first conv layer
        first_conv = None
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break

        if first_conv is None:
            raise ValueError("Could not find first Conv2d in encoder")

        if first_conv.in_channels == new_in_channels:
            return  # No adaptation needed

        # Replace with adapted conv
        old_weight = first_conv.weight.data  # (out_ch, old_in_ch, k, k)
        out_ch, old_in_ch, kH, kW = old_weight.shape

        new_conv = nn.Conv2d(
            new_in_channels,
            out_ch,
            kernel_size=(kH, kW),
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Initialize new channels by replicating averaged pretrained weights
        with torch.no_grad():
            new_weight = new_conv.weight.data
            n_repeat = new_in_channels // old_in_ch
            n_remain = new_in_channels % old_in_ch

            for i in range(n_repeat):
                new_weight[:, i * old_in_ch : (i + 1) * old_in_ch] = old_weight
            if n_remain > 0:
                new_weight[:, n_repeat * old_in_ch :] = old_weight[:, :n_remain]

            if first_conv.bias is not None:
                new_conv.bias.data.copy_(first_conv.bias.data)

        # Replace the conv in the encoder
        def _replace_first_conv(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
            setattr(parent, child_name, new_module)

        found = False
        for name, module in encoder.named_modules():
            if module is first_conv:
                # Navigate to parent
                parts = name.split(".")
                parent = encoder
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                _replace_first_conv(parent, parts[-1], new_conv)
                found = True
                break

        if not found:
            raise RuntimeError("Failed to replace first conv in encoder")

        logger.info(
            "Adapted encoder first conv: %d → %d input channels",
            old_in_ch, new_in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FloodUNet(nn.Module):
    """
    U-Net++ with EfficientNet-B7 encoder for flood detection.
    Input: 5 channels [B03, B08, B11, VV, VH]
    Output: 3 classes [dry_land, permanent_water, flooded]
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 3, encoder_name: str = "efficientnet-b7"):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes

        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch is required for FloodUNet. "
                "Install: pip install segmentation-models-pytorch"
            )

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        # The smp model already handles in_channels adaptation internally,
        # but we log it for transparency.
        logger.info(
            "FloodUNet initialized: encoder=%s, in_channels=%d, classes=%d",
            encoder_name, in_channels, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            probs = self.predict(x)
            return probs.argmax(dim=1)


class FloodUNetS2Only(nn.Module):
    """
    U-Net++ with EfficientNet-B7 encoder for optical-only flood detection.
    Input: 3 channels [B03, B08, B11]
    Output: 3 classes [dry_land, permanent_water, flooded]
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 3, encoder_name: str = "efficientnet-b7"):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes

        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch is required for FloodUNetS2Only. "
                "Install: pip install segmentation-models-pytorch"
            )

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        logger.info(
            "FloodUNetS2Only initialized: encoder=%s, in_channels=%d, classes=%d",
            encoder_name, in_channels, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = self.predict(x)
            return probs.argmax(dim=1)


def build_flood_model(
    use_sar: bool = False,
    encoder_name: str = "efficientnet-b7",
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to build the appropriate flood model.

    Args:
        use_sar: If True, build 5-channel S2+S1 model. Otherwise 3-channel S2-only.
        encoder_name: Encoder backbone name (must be supported by smp).
        weights_path: Optional path to load pretrained weights from.

    Returns:
        Initialized flood model.
    """
    if use_sar:
        model = FloodUNet(in_channels=5, num_classes=3, encoder_name=encoder_name)
    else:
        model = FloodUNetS2Only(in_channels=3, num_classes=3, encoder_name=encoder_name)

    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        model_state = state.get("model_state_dict", state)
        model.load_state_dict(model_state, strict=False)
        logger.info("Loaded flood model weights from %s", weights_path)

    return model
