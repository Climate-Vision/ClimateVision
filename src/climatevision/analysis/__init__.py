"""
ClimateVision Analysis Module

Provides extensible climate analysis types including:
- Deforestation detection
- Arctic ice melting monitoring
- Flood detection
- Drought monitoring
- Wildfire detection

Usage:
    from climatevision.analysis import get_analysis_type, list_analysis_types
    
    # Get a specific analysis type
    deforestation = get_analysis_type("deforestation")
    result = deforestation.run_inference(image_array)
    
    # List all available types
    types = list_analysis_types(enabled_only=True)
"""

from climatevision.analysis.registry import (
    AnalysisTypeRegistry,
    get_analysis_type,
    list_analysis_types,
    register_analysis_type,
)
from climatevision.analysis.base import (
    BaseAnalysisType,
    AnalysisResult,
    Alert,
)

__all__ = [
    # Registry functions
    "get_analysis_type",
    "list_analysis_types",
    "register_analysis_type",
    "AnalysisTypeRegistry",
    # Base classes
    "BaseAnalysisType",
    "AnalysisResult",
    "Alert",
]
