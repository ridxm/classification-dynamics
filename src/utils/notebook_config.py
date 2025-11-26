"""Notebook configuration loader utility.

This module provides easy access to YAML configuration files for notebooks.
All hardcoded parameters should be moved to config files and accessed via this utility.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

# Try to import OmegaConf, but fall back to dict if not available
try:
    from omegaconf import OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False


class NotebookConfig:
    """Configuration loader for notebooks.
    
    Loads and merges default and system-specific configuration files.
    Provides dot-notation access to configuration values.
    """
    
    def __init__(self, system_name: str, config_dir: str = None):
        """Initialize configuration loader.
        
        Args:
            system_name: Name of the system (e.g., 'mountain_car', 'pendulum_lqr')
            config_dir: Path to configs directory (default: '../../configs')
        """
        if config_dir is None:
            # Default: assume notebook is in notebooks/<system>/ and configs is at root
            config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            config_dir = Path(config_dir)
        
        self.config_dir = config_dir
        self.system_name = system_name
        
        # Load default notebook config
        defaults_file = config_dir / "notebook" / "analysis_defaults.yaml"
        defaults = {}
        if defaults_file.exists():
            with open(defaults_file, 'r') as f:
                defaults = yaml.safe_load(f) or {}
        
        # Load system-specific config
        system_file = config_dir / "notebook" / f"{system_name}_viz.yaml"
        system_config = {}
        if system_file.exists():
            with open(system_file, 'r') as f:
                system_config = yaml.safe_load(f) or {}
        
        # Merge configs (system-specific overrides defaults)
        self.config = self._deep_merge(defaults, system_config)
        
        # Convert to OmegaConf for dot notation access (or use dict)
        if HAS_OMEGACONF:
            self.cfg = OmegaConf.create(self.config)
        else:
            self.cfg = self.config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Access config value using dot notation.
        
        Example:
            cfg = NotebookConfig('mountain_car')
            figsize = cfg['viz.figsize']
            checkpoint = cfg['data.checkpoint_path']
        """
        keys = key.split('.')
        value = self.cfg
        for k in keys:
            if HAS_OMEGACONF:
                value = value[k]
            else:
                value = value[k]
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default if not found.
        
        Example:
            checkpoint = cfg.get('data.checkpoint_path', 'default.ckpt')
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def __repr__(self) -> str:
        return f"NotebookConfig(system='{self.system_name}')"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        if HAS_OMEGACONF:
            return OmegaConf.to_container(self.cfg, resolve=True)
        else:
            return self.config


