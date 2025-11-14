"""
Shared utilities for loading and resolving configuration files.
"""

import os
import yaml
from pathlib import Path

def load_yaml_file(file_path):
    """Load and parse a YAML file."""
    if not Path(file_path).is_file():
        return None
    
    with open(file_path, 'r') as f:
        content = os.path.expandvars(f.read())
        return yaml.safe_load(content) or {}

def load_models_registry(registry_path="configs/models.yaml"):
    """Load the models registry from YAML file."""
    data = load_yaml_file(registry_path)
    return data.get("models", {}) if data else {}

def load_providers_registry(registry_path="configs/providers.yaml"):
    """Load the providers registry from YAML file."""
    data = load_yaml_file(registry_path)
    return data.get("providers", {}) if data else {}

def load_service_registry_raw(registry_path="configs/services.yaml"):
    """Load the service registry without resolving model/provider references."""
    data = load_yaml_file(registry_path)
    return data.get("services", {}) if data else {}

