"""
Configuration management for ClimateVision
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Base configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models_pretrained"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CONFIG_DIR = PROJECT_ROOT / "config"
    
    # Data subdirectories
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SATELLITE_DATA_DIR = DATA_DIR / "satellite"
    
    # Model settings
    MODEL_INPUT_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Sentinel-2 bands (13 bands)
    SENTINEL2_BANDS = [
        "B01", "B02", "B03", "B04",  # Coastal, Blue, Green, Red
        "B05", "B06", "B07", "B08",  # Red Edge bands, NIR
        "B8A", "B09", "B10", "B11", "B12"  # Narrow NIR, Water vapor, SWIR
    ]
    
    # Band indices for vegetation
    RGB_BANDS = ["B04", "B03", "B02"]  # Red, Green, Blue
    NIR_BANDS = ["B08", "B8A"]  # Near-infrared
    SWIR_BANDS = ["B11", "B12"]  # Short-wave infrared
    
    # Training settings
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Inference settings
    CONFIDENCE_THRESHOLD = 0.7
    MIN_DEFORESTATION_AREA_HA = 0.5  # Minimum 0.5 hectares to trigger alert
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.SATELLITE_DATA_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


# Create directories on import
Config.create_directories()
