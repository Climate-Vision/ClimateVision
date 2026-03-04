"""
Training script for forest segmentation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import argparse
from tqdm import tqdm
import json

from climatevision.models.unet import create_unet


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in segmentation.
    
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        """
        # Convert to probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=inputs.shape[1]
        ).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss"""
    
    def __init__(self, focal_weight: float = 0.5):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * dice


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: (B, H, W) - predicted class indices
        targets: (B, H, W) - ground truth class indices
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for easier calculation
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()
    
    # Calculate metrics (assuming class 1 is forest)
    tp = ((pred_np == 1) & (target_np == 1)).sum()
    fp = ((pred_np == 1) & (target_np == 0)).sum()
    fn = ((pred_np == 0) & (target_np == 1)).sum()
    tn = ((pred_np == 0) & (target_np == 0)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        predictions = torch.argmax(outputs, dim=1)
        metrics = compute_metrics(predictions, masks)
        all_metrics.append(metrics)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'f1': f'{metrics["f1_score"]:.4f}'
        })
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            metrics = compute_metrics(predictions, masks)
            all_metrics.append(metrics)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = 'models',
    checkpoint_interval: int = 5
):
    """
    Train the segmentation model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N epochs
    """
    # Setup
    model = model.to(device)
    criterion = CombinedLoss(focal_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    
    print(f"Starting training on {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train'].append(train_metrics)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history['val'].append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"F1: {train_metrics['f1_score']:.4f} | "
              f"IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"F1: {val_metrics['f1_score']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1_score'],
            }
            torch.save(checkpoint, save_path / 'best_model.pth')
            print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }
            torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training completed! Best val_loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_path}")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train forest segmentation model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create model
    model = create_unet(in_channels=4, num_classes=2)
    
    print("Note: You need to implement your dataset loader.")
    print("See docs/training_guide.md for instructions on preparing your data.")
    print("\nExample dataset structure:")
    print("  data/")
    print("    train/")
    print("      images/  # Satellite images")
    print("      masks/   # Ground truth masks")
    print("    val/")
    print("      images/")
    print("      masks/")
