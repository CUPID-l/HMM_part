"""
Configuration management utilities.

Handles loading and validating YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
import logging
from string import Template

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage system configuration from YAML files"""
    
    DEFAULT_CONFIG_PATH = Path("config/default.yaml")
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config file (default: config/default.yaml)
        """
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Create one by copying config/default.yaml"
            )
        
        logger.info(f"Loading configuration from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_text = f.read()
        
        # Substitute environment variables
        config_text = self._substitute_env_vars(config_text)
        
        # Parse YAML
        config = yaml.safe_load(config_text)
        
        # Validate configuration
        self._validate_config(config)
        
        logger.info("Configuration loaded successfully")
        return config
    
    def _substitute_env_vars(self, text: str) -> str:
        """
        Substitute environment variables in config text.
        
        Supports syntax: ${VAR_NAME} or ${VAR_NAME:default_value}
        
        Args:
            text: Configuration text with placeholders
            
        Returns:
            Text with environment variables substituted
        """
        # Simple substitution for ${VAR_NAME} patterns
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            
            # Check for default value: ${VAR:default}
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = var_expr.strip()
                value = os.getenv(var_name)
                if value is None:
                    logger.warning(f"Environment variable {var_name} not set")
                    return f"${{{var_name}}}"  # Keep placeholder
                return value
        
        # Replace all ${VAR} or ${VAR:default} patterns
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_env_var, text)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['data', 'features', 'model', 'training', 'backtesting']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model parameters
        model_config = config['model']
        if model_config.get('n_states', 0) < 2:
            raise ValueError("model.n_states must be >= 2")
        
        if model_config.get('window_size', 0) < 50:
            raise ValueError("model.window_size must be >= 50")
        
        if model_config.get('covariance_type') not in ['full', 'diag', 'tied']:
            raise ValueError("model.covariance_type must be 'full', 'diag', or 'tied'")
        
        # Validate feature parameters
        features_config = config['features']
        if features_config.get('rsi_period', 0) < 1:
            raise ValueError("features.rsi_period must be >= 1")
        
        if features_config.get('volatility_window', 0) < 2:
            raise ValueError("features.volatility_window must be >= 2")
        
        # Validate backtesting parameters
        backtest_config = config['backtesting']
        if backtest_config.get('initial_capital', 0) <= 0:
            raise ValueError("backtesting.initial_capital must be > 0")
        
        if not 0 <= backtest_config.get('transaction_cost', 0) < 1:
            raise ValueError("backtesting.transaction_cost must be in [0, 1)")
        
        logger.debug("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path (e.g., 'model.n_states')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('model.n_states')
            3
            >>> config.get('model.window_size')
            252
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        logger.debug(f"Set {key_path} = {value}")
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save (default: overwrite current file)
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data section configuration"""
        return self.config.get('data', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get features section configuration"""
        return self.config.get('features', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model section configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training section configuration"""
        return self.config.get('training', {})
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtesting section configuration"""
        return self.config.get('backtesting', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring section configuration"""
        return self.config.get('monitoring', {})
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


def create_default_config(output_path: str = "config/my_config.yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Where to save the config file
    """
    output_path = Path(output_path)
    default_config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config template not found: {default_config_path}")
    
    # Copy default config
    import shutil
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(default_config_path, output_path)
    
    print(f"Created configuration file: {output_path}")
    print(f"Edit this file to customize your settings, then run:")
    print(f"  python hmm_cli.py --config {output_path}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (default: config/default.yaml)
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(config_path)
    return manager.to_dict()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management utility")
    parser.add_argument('--create', type=str, metavar='PATH',
                       help='Create a new config file at specified path')
    parser.add_argument('--validate', type=str, metavar='PATH',
                       help='Validate a config file')
    parser.add_argument('--show', type=str, metavar='PATH',
                       help='Show contents of a config file')
    
    args = parser.parse_args()
    
    if args.create:
        create_default_config(args.create)
    elif args.validate:
        try:
            config = ConfigManager(args.validate)
            print(f"✅ Configuration valid: {args.validate}")
        except Exception as e:
            print(f"❌ Configuration invalid: {e}")
    elif args.show:
        config = ConfigManager(args.show)
        import json
        print(json.dumps(config.to_dict(), indent=2))
    else:
        parser.print_help()
