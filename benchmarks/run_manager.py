#!/usr/bin/env python3
"""
Run Manager for Benchmark Runs

Manages multiple runs of benchmarks with versioned outputs for datasets,
workloads, client outputs, and trace analysis.

Usage:
    # Create a new run
    python run_manager.py create --name "baseline-test"

    # List all runs
    python run_manager.py list

    # Get current run ID
    python run_manager.py current

    # Set current run
    python run_manager.py set <run-id>

    # Delete a run
    python run_manager.py delete <run-id>
"""

import os
import sys
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path

RUNS_REGISTRY_FILE = "configs/runs.yaml"
CURRENT_RUN_FILE = "configs/.current_run"

def load_runs_registry():
    """Load the runs registry."""
    if not Path(RUNS_REGISTRY_FILE).is_file():
        return {}
    
    with open(RUNS_REGISTRY_FILE, 'r') as f:
        return yaml.safe_load(f) or {}

def save_runs_registry(registry):
    """Save the runs registry."""
    Path(RUNS_REGISTRY_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY_FILE, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

def get_current_run_id():
    """Get the current run ID."""
    if Path(CURRENT_RUN_FILE).is_file():
        with open(CURRENT_RUN_FILE, 'r') as f:
            return f.read().strip()
    return None

def set_current_run_id(run_id):
    """Set the current run ID."""
    with open(CURRENT_RUN_FILE, 'w') as f:
        f.write(run_id)

def create_run(name=None, description=None, tags=None, default_overrides=None):
    """Create a new run."""
    registry = load_runs_registry()
    
    # Generate run ID (timestamp-based)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if name:
        # Sanitize name for use in paths
        safe_name = "".join(c for c in name if c.isalnum() or c in ('-', '_')).lower()
        run_id = f"{safe_name}-{timestamp}"
    else:
        run_id = f"run-{timestamp}"
    
    if run_id in registry:
        print(f"Error: Run ID '{run_id}' already exists")
        return None
    
    # Create run entry
    run_entry = {
        "id": run_id,
        "name": name or run_id,
        "description": description or "",
        "tags": tags or [],
        "created_at": datetime.now().isoformat(),
        "status": "created"
    }
    
    # Add default overrides if provided
    if default_overrides:
        if "metadata" not in run_entry:
            run_entry["metadata"] = {}
        run_entry["metadata"]["default_overrides"] = default_overrides
    
    registry[run_id] = run_entry
    save_runs_registry(registry)
    
    # Set as current run
    set_current_run_id(run_id)
    
    print(f"Created run: {run_id}")
    print(f"  Name: {name or run_id}")
    if description:
        print(f"  Description: {description}")
    if default_overrides:
        print(f"  Default overrides: {len(default_overrides)}")
        for override in default_overrides:
            print(f"    - {override}")
    print(f"  Set as current run")
    
    return run_id

def list_runs():
    """List all runs."""
    registry = load_runs_registry()
    current_id = get_current_run_id()
    
    if not registry:
        print("No runs found")
        return
    
    print("\n========== Benchmark Runs ==========")
    for run_id, run_info in sorted(registry.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        is_current = " (current)" if run_id == current_id else ""
        print(f"\n  {run_id}{is_current}:")
        print(f"    Name: {run_info.get('name', run_id)}")
        if run_info.get('description'):
            print(f"    Description: {run_info.get('description')}")
        if run_info.get('tags'):
            print(f"    Tags: {', '.join(run_info.get('tags', []))}")
        print(f"    Created: {run_info.get('created_at', 'Unknown')}")
        print(f"    Status: {run_info.get('status', 'unknown')}")
        
        # Show metadata summary
        metadata = run_info.get('metadata', {})
        services = metadata.get('services', {})
        if services:
            print(f"    Services: {len(services)}")
            for service_id, service_data in list(services.items())[:3]:  # Show first 3
                overrides = service_data.get('overrides', [])
                if overrides:
                    print(f"      - {service_id}: {len(overrides)} override(s)")
                else:
                    print(f"      - {service_id}")
            if len(services) > 3:
                print(f"      ... and {len(services) - 3} more")
    print("\n===================================\n")

def show_current_run():
    """Show current run information."""
    current_id = get_current_run_id()
    if not current_id:
        print("No current run set")
        return
    
    registry = load_runs_registry()
    if current_id not in registry:
        print(f"Error: Current run '{current_id}' not found in registry")
        return
    
    run_info = registry[current_id]
    print(f"\nCurrent Run: {current_id}")
    print(f"  Name: {run_info.get('name', current_id)}")
    if run_info.get('description'):
        print(f"  Description: {run_info.get('description')}")
    if run_info.get('tags'):
        print(f"  Tags: {', '.join(run_info.get('tags', []))}")
    print(f"  Created: {run_info.get('created_at', 'Unknown')}")
    print(f"  Status: {run_info.get('status', 'unknown')}")
    print()

def show_run_metadata(run_id):
    """Show detailed metadata for a run."""
    registry = load_runs_registry()
    if run_id not in registry:
        print(f"Error: Run '{run_id}' not found")
        return
    
    run_info = registry[run_id]
    print(f"\n{'='*60}")
    print(f"Run Metadata: {run_id}")
    print(f"{'='*60}\n")
    
    print(f"Name: {run_info.get('name', run_id)}")
    if run_info.get('description'):
        print(f"Description: {run_info.get('description')}")
    if run_info.get('tags'):
        print(f"Tags: {', '.join(run_info.get('tags', []))}")
    print(f"Created: {run_info.get('created_at', 'Unknown')}")
    if run_info.get('updated_at'):
        print(f"Updated: {run_info.get('updated_at')}")
    print(f"Status: {run_info.get('status', 'unknown')}")
    
    metadata = run_info.get('metadata', {})
    if metadata:
        # Show default overrides
        default_overrides = metadata.get('default_overrides', [])
        if default_overrides:
            print(f"\n{'='*60}")
            print("Default Overrides:")
            print(f"{'='*60}")
            for override in default_overrides:
                print(f"  - {override}")
        
        print(f"\n{'='*60}")
        print("Services Run:")
        print(f"{'='*60}")
        services = metadata.get('services', {})
        if services:
            for service_id, service_data in services.items():
                print(f"\n  {service_id}:")
                print(f"    Stage: {service_data.get('stage', 'unknown')}")
                print(f"    Timestamp: {service_data.get('timestamp', 'Unknown')}")
                if service_data.get('model'):
                    print(f"    Model: {service_data.get('model')}")
                if service_data.get('provider'):
                    print(f"    Provider: {service_data.get('provider')}")
                if service_data.get('overrides'):
                    print(f"    Overrides:")
                    for override in service_data.get('overrides', []):
                        print(f"      - {override}")
                if service_data.get('reuse_from_run'):
                    print(f"    Reused from: {service_data.get('reuse_from_run')}")
        else:
            print("  No services recorded")
    
    # Also check for metadata files in output directories
    print(f"\n{'='*60}")
    print("Metadata Files:")
    print(f"{'='*60}")
    try:
        from run_outputs import discover_run_outputs
        outputs = discover_run_outputs(run_id)
        for output_type, info in outputs.items():
            if output_type in ["client_outputs", "trace_analysis"] and info.get("paths"):
                for path in info["paths"]:
                    metadata_file = Path(path) / "run_metadata.yaml"
                    if metadata_file.exists():
                        print(f"  ✓ {output_type}: {metadata_file}")
                    else:
                        print(f"  ✗ {output_type}: {metadata_file} (not found)")
    except ImportError:
        pass
    
    print()

def get_run_default_overrides(run_id):
    """Get default overrides for a run."""
    registry = load_runs_registry()
    if run_id not in registry:
        return []
    
    metadata = registry[run_id].get('metadata', {})
    return metadata.get('default_overrides', [])

def set_run_default_overrides(run_id, overrides):
    """Set default overrides for a run."""
    registry = load_runs_registry()
    if run_id not in registry:
        print(f"Error: Run '{run_id}' not found")
        return False
    
    update_run_metadata(run_id, {"default_overrides": overrides})
    print(f"Set default overrides for run: {run_id}")
    for override in overrides:
        print(f"  - {override}")
    return True

def set_current_run(run_id):
    """Set the current run."""
    registry = load_runs_registry()
    if run_id not in registry:
        print(f"Error: Run '{run_id}' not found")
        return False
    
    set_current_run_id(run_id)
    print(f"Set current run to: {run_id}")
    return True

def delete_run(run_id):
    """Delete a run from the registry."""
    registry = load_runs_registry()
    if run_id not in registry:
        print(f"Error: Run '{run_id}' not found")
        return False
    
    current_id = get_current_run_id()
    if run_id == current_id:
        # Clear current run if deleting it
        if Path(CURRENT_RUN_FILE).is_file():
            Path(CURRENT_RUN_FILE).unlink()
    
    del registry[run_id]
    save_runs_registry(registry)
    print(f"Deleted run: {run_id}")
    return True

def update_run_status(run_id, status):
    """Update run status."""
    registry = load_runs_registry()
    if run_id in registry:
        registry[run_id]["status"] = status
        save_runs_registry(registry)

def update_run_metadata(run_id, metadata_updates):
    """Update run metadata (overrides, services, etc.)."""
    registry = load_runs_registry()
    if run_id in registry:
        if "metadata" not in registry[run_id]:
            registry[run_id]["metadata"] = {}
        # Deep merge metadata, especially for services dict
        current_metadata = registry[run_id].get("metadata", {})
        if "services" in metadata_updates:
            if "services" not in current_metadata:
                current_metadata["services"] = {}
            current_metadata["services"].update(metadata_updates["services"])
            # Remove services from updates to avoid overwriting
            other_updates = {k: v for k, v in metadata_updates.items() if k != "services"}
            current_metadata.update(other_updates)
        else:
            current_metadata.update(metadata_updates)
        registry[run_id]["metadata"] = current_metadata
        registry[run_id]["updated_at"] = datetime.now().isoformat()
        save_runs_registry(registry)

def save_run_metadata_file(run_id, output_dir, metadata):
    """Save run metadata to a file in the output directory."""
    metadata_file = Path(output_dir) / "run_metadata.yaml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    return str(metadata_file)

def get_run_output_paths(run_id, base_paths):
    """
    Generate run-specific output paths.
    
    Args:
        run_id: Run identifier
        base_paths: Dict with keys: dataset_dir, workload_dir, client_output, trace_output
    
    Returns:
        Dict with run-specific paths
    """
    run_paths = {}
    for key, base_path in base_paths.items():
        if not base_path:
            continue
        
        # Add run_id to path
        path = Path(base_path)
        # Insert run_id before the last component or append
        if path.name:
            run_paths[key] = str(path.parent / f"{path.name}_{run_id}")
        else:
            run_paths[key] = f"{base_path}_{run_id}"
    
    return run_paths

def main():
    parser = argparse.ArgumentParser(description="Manage benchmark runs")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create run
    create_parser = subparsers.add_parser("create", help="Create a new run")
    create_parser.add_argument("--name", type=str, help="Run name")
    create_parser.add_argument("--description", type=str, help="Run description")
    create_parser.add_argument("--tags", type=str, nargs="+", help="Run tags")
    create_parser.add_argument("--default-override", type=str, action="append", dest="default_overrides",
                              help="Set default override for this run (can be used multiple times, e.g., --default-override key=value)")
    
    # List runs
    subparsers.add_parser("list", help="List all runs")
    
    # Show current run
    subparsers.add_parser("current", help="Show current run")
    
    # Set current run
    set_parser = subparsers.add_parser("set", help="Set current run")
    set_parser.add_argument("run_id", type=str, help="Run ID")
    
    # Delete run
    delete_parser = subparsers.add_parser("delete", help="Delete a run")
    delete_parser.add_argument("run_id", type=str, help="Run ID")
    
    # Show metadata
    metadata_parser = subparsers.add_parser("metadata", help="Show detailed metadata for a run")
    metadata_parser.add_argument("run_id", type=str, nargs="?", help="Run ID (default: current run)")
    
    # Set default overrides
    set_overrides_parser = subparsers.add_parser("set-overrides", help="Set default overrides for a run")
    set_overrides_parser.add_argument("run_id", type=str, nargs="?", help="Run ID (default: current run)")
    set_overrides_parser.add_argument("--override", type=str, action="append", dest="overrides",
                                     help="Override to set (can be used multiple times, e.g., --override key=value)")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_run(args.name, args.description, args.tags, args.default_overrides)
    elif args.command == "list":
        list_runs()
    elif args.command == "current":
        show_current_run()
    elif args.command == "set":
        set_current_run(args.run_id)
    elif args.command == "delete":
        delete_run(args.run_id)
    elif args.command == "metadata":
        run_id = args.run_id or get_current_run_id()
        if not run_id:
            print("Error: No run ID specified and no current run set")
            sys.exit(1)
        show_run_metadata(run_id)
    elif args.command == "set-overrides":
        run_id = args.run_id or get_current_run_id()
        if not run_id:
            print("Error: No run ID specified and no current run set")
            sys.exit(1)
        if not args.overrides:
            print("Error: No overrides specified. Use --override key=value")
            sys.exit(1)
        set_run_default_overrides(run_id, args.overrides)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

