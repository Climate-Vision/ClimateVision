"""
Evaluation metrics for segmentation and change detection models
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from typing import Tuple, Dict


def calculate_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred: Predicted segmentation mask (H, W) with class indices
        target: Ground truth segmentation mask (H, W) with class indices
        num_classes: Number of classes
    
    Returns:
        IoU for each class
    """
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')  # Avoid division by zero
        else:
            iou = intersection / union
        ious.append(iou)
    
    return np.array(ious)


def calculate_dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient (F1 score for segmentation)
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def calculate_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
    
    Returns:
        Pixel accuracy
    """
    correct = (pred == target).sum()
    total = pred.size
    return correct / total


def calculate_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics
    
    Args:
        pred: Predicted segmentation logits (N, C, H, W) or probabilities
        target: Ground truth masks (N, H, W) with class indices
    
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to class indices
    if pred.dim() == 4:  # (N, C, H, W)
        pred_classes = torch.argmax(pred, dim=1)
    else:
        pred_classes = pred
    
    # Move to CPU and convert to numpy
    pred_np = pred_classes.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Calculate metrics
    iou = calculate_iou(pred_np, target_np)
    dice = calculate_dice_coefficient((pred_np == 1).astype(float), (target_np == 1).astype(float))
    pixel_acc = calculate_pixel_accuracy(pred_np, target_np)
    
    # Calculate F1, precision, recall for forest class (class 1)
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    f1 = f1_score(target_flat, pred_flat, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(target_flat, pred_flat, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(target_flat, pred_flat, average='binary', pos_label=1, zero_division=0)
    
    return {
        'iou_mean': np.nanmean(iou),
        'iou_forest': iou[1] if len(iou) > 1 else np.nan,
        'dice_coefficient': dice,
        'pixel_accuracy': pixel_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
    }


def calculate_change_detection_metrics(pred_change: np.ndarray, target_change: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for change detection tasks
    
    Args:
        pred_change: Predicted change map (binary: 0=no change, 1=change)
        target_change: Ground truth change map
    
    Returns:
        Dictionary of change detection metrics
    """
    pred_flat = pred_change.flatten()
    target_flat = target_change.flatten()
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Kappa coefficient
    po = (tp + tn) / (tp + tn + fp + fn)  # observed agreement
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / ((tp + tn + fp + fn) ** 2)  # expected agreement
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
    }


class DiceLoss(torch.nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, C, H, W)
            target: Ground truth masks (N, H, W) with class indices
        
        Returns:
            Dice loss
        """
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - Dice as loss
        return 1 - dice.mean()


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, C, H, W)
            target: Ground truth masks (N, H, W) with class indices
        
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()
