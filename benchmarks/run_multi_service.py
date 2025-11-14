#!/usr/bin/env python3
"""
Multi-Service Benchmark Runner

This script allows you to run benchmarks across multiple services defined in
configs/services.yaml without manually editing multiple config files.

Usage:
    # Run all services
    python run_multi_service.py --stage all

    # Run specific services
    python run_multi_service.py --stage client --services openai-api openrouter-cerebras

    # Run with additional overrides
    python run_multi_service.py --stage all --override client_max_requests=100

    # List available services
    python run_multi_service.py --list-services
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add current directory to path to import benchmark module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import BenchmarkRunner, deep_merge
from config_utils import (
    load_models_registry,
    load_providers_registry,
    load_service_registry_raw
)

logging.basicConfig(level=logging.INFO)

def get_current_run_id():
    """Get the current run ID from run manager."""
    try:
        from run_manager import get_current_run_id as _get_current_run_id
        return _get_current_run_id()
    except ImportError:
        return None

def create_or_get_run(run_name=None, default_overrides=None):
    """Create a new run or get current run ID."""
    try:
        from run_manager import create_run, get_current_run_id
        current_id = get_current_run_id()
        if current_id:
            return current_id
        if run_name:
            return create_run(name=run_name, default_overrides=default_overrides)
        else:
            # Auto-create a run with timestamp
            return create_run(default_overrides=default_overrides)
    except ImportError:
        return None

def load_service_registry(registry_path="configs/services.yaml", models_registry=None, providers_registry=None):
    """Load the service registry and resolve model/provider references."""
    services = load_service_registry_raw(registry_path)
    if not services:
        return {}
    
    # Load models and providers if not provided
    if models_registry is None:
        models_registry = load_models_registry()
    if providers_registry is None:
        providers_registry = load_providers_registry()
    
    # Resolve model and provider references for each service
    resolved_services = {}
    for service_id, service_config in services.items():
        resolved_config = service_config.copy()
        
        # Resolve model reference
        model_id = service_config.get("model")
        if model_id:
            if model_id not in models_registry:
                logging.warning(f"Model '{model_id}' not found in models registry for service '{service_id}'")
            else:
                model_config = models_registry[model_id]
                # Merge model config into service overrides
                model_overrides = model_config.get("overrides", {})
                # Add model-specific fields
                if "tokenizer" in model_config:
                    model_overrides["tokenizer"] = model_config["tokenizer"]
                if "target_model" in model_config:
                    model_overrides["target_model"] = model_config["target_model"]
                if "hf_token" in model_config:
                    model_overrides["hf_token"] = model_config["hf_token"]
                if "dataset_dir" in model_config:
                    model_overrides["dataset_dir"] = model_config["dataset_dir"]
                if "workload_dir" in model_config:
                    model_overrides["workload_dir"] = model_config["workload_dir"]
                
                # Merge model overrides into service overrides
                existing_overrides = resolved_config.get("overrides", {})
                resolved_config["overrides"] = deep_merge(existing_overrides, model_overrides)
        
        # Resolve provider reference
        provider_id = service_config.get("provider")
        if provider_id:
            if provider_id not in providers_registry:
                logging.warning(f"Provider '{provider_id}' not found in providers registry for service '{service_id}'")
            else:
                provider_config = providers_registry[provider_id]
                # Merge provider config into service overrides
                provider_overrides = provider_config.get("overrides", {})
                # Add provider-specific fields
                if "endpoint" in provider_config:
                    provider_overrides["endpoint"] = provider_config["endpoint"]
                if "api_key" in provider_config:
                    provider_overrides["api_key"] = provider_config["api_key"]
                if "provider" in provider_config:
                    provider_overrides["provider"] = provider_config["provider"]
                if "openrouter_provider_config" in provider_config:
                    provider_overrides["openrouter_provider_config"] = provider_config["openrouter_provider_config"]
                # Merge workload_configs and duration_ms if present
                if "workload_configs" in provider_config:
                    existing_workload = resolved_config.get("overrides", {}).get("workload_configs", {})
                    provider_overrides["workload_configs"] = deep_merge(existing_workload, provider_config["workload_configs"])
                if "duration_ms" in provider_config:
                    provider_overrides["duration_ms"] = provider_config["duration_ms"]
                
                # Merge provider overrides into service overrides
                existing_overrides = resolved_config.get("overrides", {})
                resolved_config["overrides"] = deep_merge(existing_overrides, provider_overrides)
        
        resolved_services[service_id] = resolved_config
    
    return resolved_services

def list_services(registry, models_registry=None, providers_registry=None):
    """List all available services with model and provider info."""
    if models_registry is None:
        models_registry = load_models_registry()
    if providers_registry is None:
        providers_registry = load_providers_registry()
    
    print("\n========== Available Services ==========")
    for service_id, service_config in registry.items():
        name = service_config.get("name", service_id)
        description = service_config.get("description", "")
        model_id = service_config.get("model", "N/A")
        provider_id = service_config.get("provider", "N/A")
        
        model_name = models_registry.get(model_id, {}).get("name", model_id) if model_id != "N/A" else "N/A"
        provider_name = providers_registry.get(provider_id, {}).get("name", provider_id) if provider_id != "N/A" else "N/A"
        
        print(f"\n  {service_id}:")
        print(f"    Name: {name}")
        if description:
            print(f"    Description: {description}")
        print(f"    Model: {model_name} ({model_id})")
        print(f"    Provider: {provider_name} ({provider_id})")
    print("\n========================================\n")

def load_base_config(base_path):
    """Load base configuration."""
    from config_utils import load_yaml_file
    return load_yaml_file(base_path) or {}

def run_benchmark_for_service(service_id, service_config, stage, overrides=None, run_id=None, reuse_from_run=None):
    """Run benchmark for a single service."""
    print(f"\n{'='*60}")
    print(f"Running benchmark for: {service_config.get('name', service_id)}")
    if run_id:
        print(f"Run ID: {run_id}")
    if reuse_from_run:
        print(f"Reusing outputs from: {reuse_from_run}")
    print(f"{'='*60}\n")
    
    base_path = service_config.get("base", "configs/base.yaml")
    service_overrides = service_config.get("overrides", {})
    
    # Combine service overrides with command-line overrides
    all_overrides = overrides or []
    
    # Create BenchmarkRunner with service config, run_id, and reuse_from_run
    runner = BenchmarkRunner(
        config_base=base_path,
        overrides=all_overrides,
        service_config=service_overrides,
        run_id=run_id,
        reuse_from_run=reuse_from_run
    )
    
    # Save run metadata if run_id is provided
    if run_id:
        try:
            from run_manager import update_run_metadata, save_run_metadata_file
            import yaml
            from datetime import datetime
            
            # Get run info from registry
            from run_manager import load_runs_registry
            registry = load_runs_registry()
            run_info = registry.get(run_id, {})
            
            # Prepare metadata
            metadata = {
                "run_id": run_id,
                "run_name": run_info.get("name", run_id),
                "run_description": run_info.get("description", ""),
                "run_tags": run_info.get("tags", []),
                "run_created_at": run_info.get("created_at"),
                "service_id": service_id,
                "service_name": service_config.get("name", service_id),
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                "base_config": base_path,
                "model": service_config.get("model"),
                "provider": service_config.get("provider"),
                "service_overrides": service_overrides,
                "command_line_overrides": all_overrides,
                "reuse_from_run": reuse_from_run,
                "final_config": {
                    "dataset_dir": runner.config.get("dataset_dir"),
                    "workload_dir": runner.config.get("workload_dir"),
                    "client_output": runner.config.get("client_output"),
                    "trace_output": runner.config.get("trace_output"),
                }
            }
            
            # Update run registry
            update_run_metadata(run_id, {
                "services": {service_id: {
                    "stage": stage,
                    "overrides": all_overrides,
                    "reuse_from_run": reuse_from_run,
                    "timestamp": datetime.now().isoformat(),
                    "model": service_config.get("model"),
                    "provider": service_config.get("provider"),
                }}
            })
            
            # Save metadata file in output directories
            for output_key in ["client_output", "trace_output"]:
                if output_key in runner.config:
                    output_dir = runner.config[output_key]
                    metadata_file = save_run_metadata_file(run_id, output_dir, metadata)
                    logging.info(f"Saved run metadata to: {metadata_file}")
        except Exception as e:
            logging.warning(f"Failed to save run metadata: {e}")
    
    # Run the specified stage
    if stage == "all":
        runner.generate_dataset()
        runner.generate_workload()
        runner.run_client()
        runner.run_analysis()
    elif stage == "dataset":
        runner.generate_dataset()
    elif stage == "workload":
        runner.generate_workload()
    elif stage == "client":
        runner.run_client()
    elif stage == "analysis":
        runner.run_analysis()
    else:
        logging.error(f"Unknown stage: {stage}")
        return False
    
    print(f"\n✓ Completed {stage} for {service_id}\n")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks across multiple services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=False,
        default="all",
        choices=["all", "dataset", "workload", "client", "analysis"],
        help="Benchmark stage to run"
    )
    parser.add_argument(
        "--services",
        type=str,
        nargs="+",
        help="Specific services to run (default: all services)"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/services.yaml",
        help="Path to service registry YAML file"
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override config values (can be used multiple times, e.g., --override key=value)"
    )
    parser.add_argument(
        "--list-services",
        action="store_true",
        help="List all available services and exit"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run services in parallel (experimental)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to use for this benchmark (default: use current run or auto-create)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Create a new run with this name (if no current run exists)"
    )
    parser.add_argument(
        "--no-run-id",
        action="store_true",
        help="Don't use run management (use original output paths)"
    )
    parser.add_argument(
        "--reuse-from-run",
        type=str,
        default=None,
        help="Reuse outputs from another run (format: 'run-id' or 'run-id:dataset,workload' or 'run-id:all')"
    )
    
    args = parser.parse_args()
    
    # Load models and providers registries
    models_registry = load_models_registry()
    providers_registry = load_providers_registry()
    
    # Load service registry (with model/provider resolution)
    registry = load_service_registry(args.registry, models_registry, providers_registry)
    
    if args.list_services:
        list_services(registry, models_registry, providers_registry)
        return
    
    if not registry:
        logging.error("No services found in registry")
        sys.exit(1)
    
    # Determine which services to run
    if args.services:
        services_to_run = {sid: registry[sid] for sid in args.services if sid in registry}
        if not services_to_run:
            logging.error(f"None of the specified services found: {args.services}")
            logging.info("Available services:")
            list_services(registry)
            sys.exit(1)
        missing = set(args.services) - set(services_to_run.keys())
        if missing:
            logging.warning(f"Services not found: {missing}")
    else:
        services_to_run = registry
    
    # Determine run_id
    run_id = None
    if not args.no_run_id:
        if args.run_id:
            run_id = args.run_id
        else:
            # Get or create run
            run_id = get_current_run_id()
            if not run_id and args.run_name:
                run_id = create_or_get_run(run_name=args.run_name)
            elif not run_id:
                # Auto-create a run
                run_id = create_or_get_run()
    
    # Get default overrides for the run
    run_default_overrides = []
    if run_id:
        try:
            from run_manager import get_run_default_overrides
            run_default_overrides = get_run_default_overrides(run_id)
            if run_default_overrides:
                logging.info(f"Using {len(run_default_overrides)} default override(s) from run: {run_id}")
        except Exception as e:
            logging.warning(f"Failed to load default overrides: {e}")
    
    # Combine default overrides with command-line overrides (command-line takes precedence)
    all_overrides = []
    if run_default_overrides:
        all_overrides.extend(run_default_overrides)
    if args.override:
        all_overrides.extend(args.override)
    
    # Use combined overrides
    final_overrides = all_overrides if all_overrides else None
    
    print(f"\n{'='*60}")
    print(f"Multi-Service Benchmark Runner")
    print(f"Stage: {args.stage}")
    print(f"Services: {len(services_to_run)}")
    if run_id:
        print(f"Run ID: {run_id}")
    if run_default_overrides:
        print(f"Default Overrides: {len(run_default_overrides)}")
        for override in run_default_overrides:
            print(f"  - {override}")
    if args.override:
        print(f"Command-line Overrides: {len(args.override)}")
        for override in args.override:
            print(f"  - {override}")
    print(f"{'='*60}\n")
    
    # Run benchmarks
    results = {}
    for service_id, service_config in services_to_run.items():
        try:
            success = run_benchmark_for_service(
                service_id,
                service_config,
                args.stage,
                final_overrides,
                run_id=run_id,
                reuse_from_run=args.reuse_from_run
            )
            results[service_id] = "success" if success else "failed"
        except Exception as e:
            logging.error(f"Error running benchmark for {service_id}: {e}")
            results[service_id] = "error"
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for service_id, status in results.items():
        status_symbol = "✓" if status == "success" else "✗"
        print(f"  {status_symbol} {service_id}: {status}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

