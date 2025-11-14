#!/usr/bin/env python3
"""
Generate individual config files from service registry.

This script generates standalone YAML config files from the service registry,
useful if you need individual files for compatibility or manual editing.

Usage:
    # Generate config for a specific service
    python generate_config.py openai-api

    # Generate configs for multiple services
    python generate_config.py openai-api openrouter-cerebras

    # Generate all configs
    python generate_config.py --all
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import deep_merge
from config_utils import (
    load_models_registry,
    load_providers_registry,
    load_service_registry_raw
)

# Import resolution function from run_multi_service
from run_multi_service import load_service_registry

def load_base_config(base_path):
    """Load base configuration."""
    if not Path(base_path).is_file():
        print(f"Error: Base config not found: {base_path}")
        return None
    
    with open(base_path, 'r') as f:
        content = os.path.expandvars(f.read())
        return yaml.safe_load(content) or {}

# load_service_registry is now imported from run_multi_service

def generate_config(service_id, service_config, output_dir="configs/generated"):
    """Generate a config file for a service."""
    base_path = service_config.get("base", "configs/base.yaml")
    service_overrides = service_config.get("overrides", {})
    
    # Load base config
    base_config = load_base_config(base_path)
    if base_config is None:
        return False
    
    # Merge with service overrides
    merged_config = deep_merge(base_config, service_overrides)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = Path(output_dir) / f"config-{service_id}.yaml"
    
    # Write merged config
    with open(output_file, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate config files from service registry")
    parser.add_argument(
        "services",
        nargs="*",
        help="Service IDs to generate configs for"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate configs for all services"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/services.yaml",
        help="Path to service registry"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/generated",
        help="Output directory for generated configs"
    )
    
    args = parser.parse_args()
    
    # Load models and providers registries
    models_registry = load_models_registry()
    providers_registry = load_providers_registry()
    
    # Load service registry (with model/provider resolution)
    registry = load_service_registry(args.registry, models_registry, providers_registry)
    if not registry:
        print("Error: No services found in registry")
        sys.exit(1)
    
    # Determine which services to generate
    if args.all:
        services_to_generate = list(registry.keys())
    elif args.services:
        services_to_generate = [s for s in args.services if s in registry]
        missing = set(args.services) - set(services_to_generate)
        if missing:
            print(f"Warning: Services not found: {missing}")
    else:
        parser.print_help()
        sys.exit(1)
    
    if not services_to_generate:
        print("No services to generate")
        sys.exit(1)
    
    # Generate configs
    print(f"Generating configs for {len(services_to_generate)} service(s)...\n")
    success_count = 0
    for service_id in services_to_generate:
        if generate_config(service_id, registry[service_id], args.output_dir):
            success_count += 1
    
    print(f"\n✓ Generated {success_count}/{len(services_to_generate)} config files")

if __name__ == "__main__":
    main()

