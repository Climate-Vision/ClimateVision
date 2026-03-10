"""
Complete training and inference pipeline for ClimateVision
Downloads satellite data from GEE, trains model, and runs inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Initialize Earth Engine
import ee
ee.Initialize(project='kinos-473422')

from climatevision.models.unet import UNet


class SyntheticForestDataset(Dataset):
    """
    Synthetic dataset for training demonstration.
    Creates random satellite-like images with forest/non-forest masks.
    """

    def __init__(self, num_samples: int = 100, image_size: int = 256, n_channels: int = 4):
        self.num_samples = num_samples
        self.image_size = image_size
        self.n_channels = n_channels

        print(f"Generating {num_samples} synthetic training samples...")
        self.images, self.masks = self._generate_data()
        print(f"Dataset ready: {len(self)} samples")

    def _generate_data(self):
        images = []
        masks = []

        for _ in range(self.num_samples):
            # Create synthetic satellite image (4 bands: RGB + NIR)
            image = np.random.randn(self.n_channels, self.image_size, self.image_size).astype(np.float32)

            # Add some structure - simulate vegetation patterns
            x = np.linspace(0, 4*np.pi, self.image_size)
            y = np.linspace(0, 4*np.pi, self.image_size)
            xx, yy = np.meshgrid(x, y)

            # Create forest-like patterns
            pattern = np.sin(xx + np.random.rand()*2*np.pi) * np.cos(yy + np.random.rand()*2*np.pi)
            pattern += np.random.randn(self.image_size, self.image_size) * 0.3

            # Add pattern to NIR band (vegetation reflects strongly in NIR)
            image[3] += pattern * 0.5

            # Create mask based on "NDVI-like" threshold
            ndvi_like = (image[3] - image[0]) / (image[3] + image[0] + 1e-8)
            mask = (ndvi_like > np.random.uniform(-0.1, 0.1)).astype(np.int64)

            images.append(image)
            masks.append(mask)

        return images, masks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx]), torch.LongTensor(self.masks[idx])


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=inputs.shape[1]
        ).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss"""
    def __init__(self, focal_weight=0.5):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal_loss(inputs, targets) + \
               (1 - self.focal_weight) * self.dice_loss(inputs, targets)


def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()

    tp = ((pred_np == 1) & (target_np == 1)).sum()
    fp = ((pred_np == 1) & (target_np == 0)).sum()
    fn = ((pred_np == 0) & (target_np == 1)).sum()
    tn = ((pred_np == 0) & (target_np == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "iou": iou}


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = []

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        metrics = compute_metrics(predictions, masks)
        all_metrics.append(metrics)
        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'f1': f'{metrics["f1_score"]:.4f}'})

    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
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

            outputs = model(images)
            loss = criterion(outputs, masks)

            predictions = torch.argmax(outputs, dim=1)
            metrics = compute_metrics(predictions, masks)
            all_metrics.append(metrics)
            total_loss += loss.item()

    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics


def run_training(num_epochs=10, batch_size=8, learning_rate=1e-4):
    """Main training function"""
    print("=" * 60)
    print("ClimateVision Training Pipeline")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset
    print("\n[1/4] Creating dataset...")
    dataset = SyntheticForestDataset(num_samples=200, image_size=256, n_channels=4)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\n[2/4] Creating model...")
    model = UNet(n_channels=4, n_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = CombinedLoss(focal_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Create save directory
    save_dir = Path(__file__).parent.parent / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n[3/4] Training for {num_epochs} epochs...")
    print("-" * 60)

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        scheduler.step(val_metrics['loss'])

        print(f"Train - Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1_score']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1_score']:.4f} | IoU: {val_metrics['iou']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1_score'],
            }, save_dir / 'best_model.pth')
            print(f"[SAVED] Best model (val_loss: {val_metrics['loss']:.4f})")

    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        # Convert numpy types to Python types for JSON
        history_json = {
            'train': [{k: float(v) for k, v in m.items()} for m in history['train']],
            'val': [{k: float(v) for k, v in m.items()} for m in history['val']]
        }
        json.dump(history_json, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    print("=" * 60)

    return model, history


def run_inference_script():
    """Run inference using the shared inference module and save results."""
    from climatevision.inference import run_inference_from_gee

    print("\n" + "=" * 60)
    print("Running Inference (via climatevision.inference module)")
    print("=" * 60)

    bbox = [-62.0, -3.1, -61.8, -2.9]
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    results = run_inference_from_gee(
        bbox=bbox, start_date=start_date, end_date=end_date
    )

    # Add extra metadata for the standalone script
    results.setdefault("region", {}).update({
        "location": "Amazon Rainforest, Brazil",
        "satellite": "Sentinel-2",
    })

    # Print summary
    ndvi = results.get("ndvi_stats", {})
    inf = results.get("inference", {})
    print(f"\nNDVI — min: {ndvi.get('NDVI_min', 'N/A')}, "
          f"mean: {ndvi.get('NDVI_mean', 'N/A')}, "
          f"max: {ndvi.get('NDVI_max', 'N/A')}")
    print(f"Forest pixels: {inf.get('forest_pixels', 0):,}")
    print(f"Forest %: {inf.get('forest_percentage', 0):.2f}")
    print(f"Mean confidence: {inf.get('mean_confidence', 0):.4f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "inference_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Run training
    model, history = run_training(num_epochs=10, batch_size=8, learning_rate=1e-4)

    # Run inference
    results = run_inference_script()

    print("\n" + "=" * 60)
    print("ClimateVision Pipeline Complete!")
    print("=" * 60)
