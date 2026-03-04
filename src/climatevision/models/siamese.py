"""
Siamese Network for change detection in satellite imagery

Compares two images from different time periods to detect changes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SiameseEncoder(nn.Module):
    """Shared encoder for Siamese network"""
    
    def __init__(self, in_channels: int = 13):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.encoder(x)


class ChangeDecoder(nn.Module):
    """Decoder for change map generation"""
    
    def __init__(self, in_channels: int = 1024):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample 2
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample 3
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Output layer (2 classes: no-change, change)
            nn.Conv2d(64, 2, kernel_size=1),
        )
    
    def forward(self, x):
        return self.decoder(x)


class SiameseNetwork(nn.Module):
    """
    Siamese Network for change detection
    
    Takes two images from different time periods and outputs a change map
    
    Args:
        in_channels: Number of input channels per image (e.g., 13 for Sentinel-2)
    """
    
    def __init__(self, in_channels: int = 13):
        super().__init__()
        self.in_channels = in_channels
        
        # Shared encoder
        self.encoder = SiameseEncoder(in_channels)
        
        # Decoder for change map
        self.decoder = ChangeDecoder(in_channels=1024)
    
    def forward(self, image_before: torch.Tensor, image_after: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image_before: Image from earlier time (N, C, H, W)
            image_after: Image from later time (N, C, H, W)
        
        Returns:
            Change map logits (N, 2, H, W)
        """
        # Extract features from both images using shared encoder
        features_before = self.encoder(image_before)
        features_after = self.encoder(image_after)
        
        # Concatenate features
        combined_features = torch.cat([features_before, features_after], dim=1)
        
        # Generate change map
        change_logits = self.decoder(combined_features)
        
        return change_logits
    
    def predict(self, image_before: torch.Tensor, image_after: torch.Tensor) -> torch.Tensor:
        """
        Get change prediction probabilities
        
        Args:
            image_before: Image from earlier time
            image_after: Image from later time
        
        Returns:
            Change probabilities (N, 2, H, W)
        """
        logits = self.forward(image_before, image_after)
        return F.softmax(logits, dim=1)
    
    def predict_binary(self, image_before: torch.Tensor, image_after: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary change map
        
        Args:
            image_before: Image from earlier time
            image_after: Image from later time
            threshold: Threshold for change detection
        
        Returns:
            Binary change map (N, H, W) - 0: no change, 1: change
        """
        probs = self.predict(image_before, image_after)
        change_prob = probs[:, 1, :, :]  # Probability of change class
        binary_change = (change_prob > threshold).long()
        return binary_change


class EarlyFusion(nn.Module):
    """
    Early fusion approach for change detection
    
    Concatenates images before encoding (simpler alternative to Siamese)
    """
    
    def __init__(self, in_channels: int = 13, n_classes: int = 2):
        super().__init__()
        
        # Encoder for concatenated images
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, n_classes, kernel_size=1),
        )
    
    def forward(self, image_before: torch.Tensor, image_after: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion
        
        Args:
            image_before: Image from earlier time (N, C, H, W)
            image_after: Image from later time (N, C, H, W)
        
        Returns:
            Change map logits (N, 2, H, W)
        """
        # Concatenate images along channel dimension
        combined = torch.cat([image_before, image_after], dim=1)
        
        # Encode and decode
        features = self.encoder(combined)
        change_logits = self.decoder(features)
        
        return change_logits
    
    def predict(self, image_before: torch.Tensor, image_after: torch.Tensor) -> torch.Tensor:
        """Get change prediction probabilities"""
        logits = self.forward(image_before, image_after)
        return F.softmax(logits, dim=1)
    
    def predict_binary(self, image_before: torch.Tensor, image_after: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary change map"""
        probs = self.predict(image_before, image_after)
        change_prob = probs[:, 1, :, :]
        binary_change = (change_prob > threshold).long()
        return binary_change
