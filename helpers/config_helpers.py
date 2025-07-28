"""
Configuration management utilities for PerToon project.
Provides functions to load and manage YAML configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")

def load_vision_config() -> Dict[str, Any]:
    """Load vision model configuration."""
    return load_config("configs/vision_config.yaml")

def load_nlp_config() -> Dict[str, Any]:
    """Load NLP model configuration."""
    return load_config("configs/nlp_config.yaml")

def load_pipeline_config() -> Dict[str, Any]:
    """Load pipeline configuration."""
    return load_config("configs/pipeline_config.yaml")

def get_model_paths(config_type: str = "vision") -> Dict[str, str]:
    """
    Get model file paths from configuration.
    
    Args:
        config_type (str): Type of configuration ("vision", "nlp", or "pipeline")
        
    Returns:
        Dict[str, str]: Dictionary of model paths
    """
    if config_type == "vision":
        config = load_vision_config()
        return config.get("paths", {})
    elif config_type == "nlp":
        config = load_nlp_config()
        return config.get("paths", {})
    elif config_type == "pipeline":
        config = load_pipeline_config()
        return config.get("io", {})
    else:
        raise ValueError(f"Unknown config type: {config_type}")

def get_device_config(config_type: str = "vision") -> Dict[str, Any]:
    """
    Get device configuration (GPU/CPU settings).
    
    Args:
        config_type (str): Type of configuration ("vision" or "nlp")
        
    Returns:
        Dict[str, Any]: Device configuration
    """
    if config_type == "vision":
        config = load_vision_config()
    elif config_type == "nlp":
        config = load_nlp_config()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return config.get("device", {"use_cuda": True})

def update_config(config_path: str, updates: Dict[str, Any]) -> None:
    """
    Update configuration file with new values.
    
    Args:
        config_path (str): Path to the configuration file
        updates (Dict[str, Any]): Dictionary of updates to apply
    """
    config = load_config(config_path)
    
    # Deep merge updates into config
    def deep_merge(dict1, dict2):
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                deep_merge(dict1[key], value)
            else:
                dict1[key] = value
    
    deep_merge(config, updates)
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that configuration contains required keys.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        required_keys (list): List of required keys (supports nested keys with dot notation)
        
    Returns:
        bool: True if all required keys are present
        
    Raises:
        KeyError: If required key is missing
    """
    for key in required_keys:
        keys = key.split('.')
        current = config
        
        for k in keys:
            if k not in current:
                raise KeyError(f"Required configuration key missing: {key}")
            current = current[k]
    
    return True

def get_generation_params() -> Dict[str, Any]:
    """Get text generation parameters from NLP config."""
    config = load_nlp_config()
    return config.get("generation", {})

def get_mood_categories() -> list:
    """Get available mood categories from NLP config."""
    config = load_nlp_config()
    return config.get("moods", ["happy", "sad", "funny", "neutral"])

def ensure_directories_exist():
    """Create necessary directories based on configuration."""
    configs = [load_vision_config(), load_nlp_config(), load_pipeline_config()]
    
    directories = set()
    for config in configs:
        # Extract directory paths from config
        if "paths" in config:
            for path in config["paths"].values():
                if isinstance(path, str) and "/" in path:
                    directories.add(os.path.dirname(path))
        
        if "io" in config:
            for path in config["io"].values():
                if isinstance(path, str) and "/" in path:
                    directories.add(path if not path.endswith("/") else path[:-1])
    
    # Create directories
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}") 