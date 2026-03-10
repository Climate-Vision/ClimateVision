"""
Analysis Type Registry

Manages registration and lookup of analysis types.
Allows dynamic registration of new analysis types.
"""

from __future__ import annotations

from typing import Optional, Type
import logging

from climatevision.analysis.base import BaseAnalysisType

logger = logging.getLogger(__name__)


class AnalysisTypeRegistry:
    """
    Registry for managing analysis types.
    
    Usage:
        registry = AnalysisTypeRegistry()
        registry.register(DeforestationAnalysis)
        
        # Get an analysis type
        analysis = registry.get("deforestation")
        
        # List all types
        all_types = registry.list_all()
    """
    
    _instance: Optional["AnalysisTypeRegistry"] = None
    
    def __new__(cls) -> "AnalysisTypeRegistry":
        """Singleton pattern - only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._types = {}
            cls._instance._initialized = False
        return cls._instance
    
    def register(
        self,
        analysis_class: Type[BaseAnalysisType],
        override: bool = False,
    ) -> None:
        """
        Register an analysis type.
        
        Args:
            analysis_class: The analysis type class to register
            override: Whether to override existing registration
        
        Raises:
            ValueError: If name is already registered and override is False
        """
        name = analysis_class.name
        
        if not name:
            raise ValueError(f"Analysis class {analysis_class} has no 'name' attribute")
        
        if name in self._types and not override:
            raise ValueError(
                f"Analysis type '{name}' is already registered. "
                f"Use override=True to replace."
            )
        
        self._types[name] = analysis_class
        logger.info(f"Registered analysis type: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister an analysis type.
        
        Args:
            name: Name of the analysis type to unregister
        
        Returns:
            True if type was unregistered, False if it wasn't registered
        """
        if name in self._types:
            del self._types[name]
            logger.info(f"Unregistered analysis type: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseAnalysisType]:
        """
        Get an instance of an analysis type.
        
        Args:
            name: Name of the analysis type
        
        Returns:
            Instance of the analysis type, or None if not found
        """
        analysis_class = self._types.get(name)
        if analysis_class:
            return analysis_class()
        return None
    
    def get_class(self, name: str) -> Optional[Type[BaseAnalysisType]]:
        """
        Get the class of an analysis type.
        
        Args:
            name: Name of the analysis type
        
        Returns:
            The analysis type class, or None if not found
        """
        return self._types.get(name)
    
    def list_all(self, enabled_only: bool = False) -> list[dict]:
        """
        List all registered analysis types.
        
        Args:
            enabled_only: If True, only return enabled types
        
        Returns:
            List of analysis type info dictionaries
        """
        result = []
        for name, analysis_class in self._types.items():
            instance = analysis_class()
            if enabled_only and not instance.enabled:
                continue
            result.append(instance.get_info())
        return result
    
    def is_registered(self, name: str) -> bool:
        """Check if an analysis type is registered."""
        return name in self._types
    
    def clear(self) -> None:
        """Clear all registered types. Use with caution."""
        self._types.clear()
        self._initialized = False
        logger.warning("Cleared all registered analysis types")


# Global registry instance
_registry = AnalysisTypeRegistry()


def register_analysis_type(
    analysis_class: Type[BaseAnalysisType],
    override: bool = False,
) -> None:
    """
    Register an analysis type with the global registry.
    
    Args:
        analysis_class: The analysis type class to register
        override: Whether to override existing registration
    """
    _registry.register(analysis_class, override)


def get_analysis_type(name: str) -> Optional[BaseAnalysisType]:
    """
    Get an analysis type from the global registry.
    
    Args:
        name: Name of the analysis type
    
    Returns:
        Instance of the analysis type, or None if not found
    """
    # Ensure built-in types are registered
    _ensure_builtins_registered()
    return _registry.get(name)


def list_analysis_types(enabled_only: bool = True) -> list[dict]:
    """
    List all registered analysis types.
    
    Args:
        enabled_only: If True, only return enabled types
    
    Returns:
        List of analysis type info dictionaries
    """
    _ensure_builtins_registered()
    return _registry.list_all(enabled_only)


def _ensure_builtins_registered() -> None:
    """Ensure built-in analysis types are registered."""
    if _registry._initialized:
        return
    
    # Import and register built-in types
    try:
        from climatevision.analysis.deforestation import DeforestationAnalysis
        _registry.register(DeforestationAnalysis, override=True)
    except ImportError as e:
        logger.warning(f"Could not import DeforestationAnalysis: {e}")
    
    try:
        from climatevision.analysis.ice_melting import IceMeltingAnalysis
        _registry.register(IceMeltingAnalysis, override=True)
    except ImportError as e:
        logger.warning(f"Could not import IceMeltingAnalysis: {e}")
    
    try:
        from climatevision.analysis.flooding import FloodingAnalysis
        _registry.register(FloodingAnalysis, override=True)
    except ImportError as e:
        logger.warning(f"Could not import FloodingAnalysis: {e}")
    
    _registry._initialized = True
