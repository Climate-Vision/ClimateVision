"""
U-Net architecture for semantic segmentation

Reference: U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation
    
    Args:
        n_channels: Number of input channels (e.g., 13 for Sentinel-2)
        n_classes: Number of output classes (e.g., 2 for forest/non-forest)
        bilinear: Use bilinear upsampling instead of transposed convolutions
    """
    
    def __init__(self, n_channels: int = 13, n_classes: int = 2, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def predict(self, x):
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor (N, C, H, W)
        
        Returns:
            Probability maps (N, n_classes, H, W)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_classes(self, x):
        """
        Get predicted class indices
        
        Args:
            x: Input tensor (N, C, H, W)
        
        Returns:
            Class predictions (N, H, W)
        """
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)


class AttentionBlock(nn.Module):
    """Attention block for Attention U-Net"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from coarser scale (N, F_g, H_g, W_g)
            x: Feature map from encoder (N, F_l, H_x, W_x)
        
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for improved segmentation
    
    Incorporates attention gates to focus on relevant features
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
    """
    
    def __init__(self, n_channels: int = 13, n_classes: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Attention gates
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Decoder
        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        x4_att = self.att1(g=x5, x=x4)
        x = self.up1(x5, x4_att)
        
        x3_att = self.att2(g=x, x=x3)
        x = self.up2(x, x3_att)
        
        x2_att = self.att3(g=x, x=x2)
        x = self.up3(x, x2_att)
        
        x1_att = self.att4(g=x, x=x1)
        x = self.up4(x, x1_att)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def predict(self, x):
        """Get prediction probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_classes(self, x):
        """Get predicted class indices"""
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)


class ResNetUNet(nn.Module):
    """
    U-Net with a pretrained ResNet encoder for transfer learning.

    Uses a ResNet-34 (or ResNet-50) backbone pretrained on ImageNet as the
    encoder and a lightweight decoder with skip connections.

    For satellite imagery with more than 3 input channels (e.g. 4 or 13),
    the first convolutional layer is adapted: the pretrained RGB weights are
    preserved and extra channels are initialised with the mean of the RGB
    weights so training starts from a reasonable point.

    Args:
        n_channels: Number of input bands (e.g. 4 for R,G,B,NIR).
        n_classes:  Number of output classes.
        backbone:   ``'resnet34'`` or ``'resnet50'``.
        pretrained: Load ImageNet weights.
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 2,
        backbone: str = "resnet34",
        pretrained: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        import torchvision.models as tv

        weights_map = {
            "resnet34": (tv.resnet34, tv.ResNet34_Weights.DEFAULT if pretrained else None),
            "resnet50": (tv.resnet50, tv.ResNet50_Weights.DEFAULT if pretrained else None),
        }
        if backbone not in weights_map:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {list(weights_map)}")

        builder, weights = weights_map[backbone]
        encoder = builder(weights=weights)

        # Adapt first conv for non-3-channel inputs
        old_conv = encoder.conv1
        if n_channels != 3:
            new_conv = nn.Conv2d(
                n_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                # Copy RGB weights and init extra channels with the RGB mean
                new_conv.weight[:, :3] = old_conv.weight
                if n_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:] = mean_w.expand_as(new_conv.weight[:, 3:])
            encoder.conv1 = new_conv

        # Encoder stages (ResNet layers)
        self.enc0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # /2
        self.pool0 = encoder.maxpool                                          # /4
        self.enc1 = encoder.layer1                                            # /4
        self.enc2 = encoder.layer2                                            # /8
        self.enc3 = encoder.layer3                                            # /16
        self.enc4 = encoder.layer4                                            # /32

        # Channel counts depend on backbone
        if backbone == "resnet34":
            enc_channels = [64, 64, 128, 256, 512]
        else:  # resnet50
            enc_channels = [64, 256, 512, 1024, 2048]

        # Decoder
        self.up4 = _DecoderBlock(enc_channels[4], enc_channels[3], 256)
        self.up3 = _DecoderBlock(256, enc_channels[2], 128)
        self.up2 = _DecoderBlock(128, enc_channels[1], 64)
        self.up1 = _DecoderBlock(64, enc_channels[0], 64)

        # Final upsample (encoder first stride is 2, pool takes to 4)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)       # /2
        p0 = self.pool0(e0)     # /4
        e1 = self.enc1(p0)      # /4
        e2 = self.enc2(e1)      # /8
        e3 = self.enc3(e2)      # /16
        e4 = self.enc4(e3)      # /32

        # Decoder with skip connections
        d4 = self.up4(e4, e3)   # /16
        d3 = self.up3(d4, e2)   # /8
        d2 = self.up2(d3, e1)   # /4
        d1 = self.up1(d2, e0)   # /2
        d0 = self.final_up(d1)  # /1
        return self.outc(d0)

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict_classes(self, x):
        return self.predict(x).argmax(dim=1)


class _DecoderBlock(nn.Module):
    """Upsample + concatenate skip + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims don't match
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x, skip], dim=1))


def get_model(model_name: str = "unet", **kwargs) -> nn.Module:
    """
    Factory function to get model by name
    
    Args:
        model_name: Name of the model ('unet', 'attention_unet', or 'resnet_unet')
        **kwargs: Additional arguments for the model
    
    Returns:
        Model instance
    """
    models = {
        'unet': UNet,
        'attention_unet': AttentionUNet,
        'resnet_unet': ResNetUNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def create_unet(**kwargs) -> UNet:
    """Convenience alias used by scripts/train.py."""
    return UNet(**kwargs)
