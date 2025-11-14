#!/usr/bin/env python3
"""
Run Outputs Manager

Manages and reuses outputs (datasets, workloads, client outputs, trace analysis)
from previous runs.

Usage:
    # List outputs for a run
    python run_outputs.py list <run-id>

    # List outputs for all runs
    python run_outputs.py list-all

    # Copy outputs from one run to another
    python run_outputs.py copy <source-run-id> <target-run-id> [--dataset] [--workload] [--client] [--trace]

    # Link outputs (symlink) from one run to another
    python run_outputs.py link <source-run-id> <target-run-id> [--dataset] [--workload] [--client] [--trace]

    # Check if outputs exist for a run
    python run_outputs.py check <run-id>
"""

import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path
from run_manager import load_runs_registry, get_current_run_id

def discover_run_outputs(run_id, config=None):
    """
    Discover what outputs exist for a given run.
    
    Returns a dict with keys: dataset, workload, client_outputs, trace_analysis
    Each value is a dict with 'path' (or 'paths' for multiple), 'exists', and 'files' keys.
    For multiple outputs, 'paths' will be a list.
    """
    outputs = {
        "dataset": {"path": None, "paths": [], "exists": False, "files": []},
        "workload": {"path": None, "paths": [], "exists": False, "files": []},
        "client_outputs": {"path": None, "paths": [], "exists": False, "files": []},
        "trace_analysis": {"path": None, "paths": [], "exists": False, "files": []}
    }
    
    if not config:
        # Try to discover from common patterns
        # Look for directories matching run_id pattern
        output_base = Path("./output")
        if not output_base.exists():
            return outputs
        
        # Find dataset directories
        dataset_dirs = list(output_base.glob(f"dataset_*_{run_id}"))
        if dataset_dirs:
            # Use first one for backward compatibility
            outputs["dataset"]["path"] = str(dataset_dirs[0])
            outputs["dataset"]["paths"] = [str(d) for d in dataset_dirs]
            outputs["dataset"]["exists"] = any(d.exists() for d in dataset_dirs)
            for dataset_dir in dataset_dirs:
                if dataset_dir.exists():
                    outputs["dataset"]["files"].extend([str(f) for f in dataset_dir.glob("*.jsonl")])
        
        # Find workload directories
        workload_dirs = []
        for workload_dir in output_base.glob(f"workload_*/**/*_{run_id}"):
            if workload_dir.is_dir():
                workload_dirs.append(workload_dir)
        if workload_dirs:
            outputs["workload"]["path"] = str(workload_dirs[0])
            outputs["workload"]["paths"] = [str(d) for d in workload_dirs]
            outputs["workload"]["exists"] = any(d.exists() for d in workload_dirs)
            for workload_dir in workload_dirs:
                if workload_dir.exists():
                    outputs["workload"]["files"].extend([str(f) for f in workload_dir.glob("*.jsonl")])
        
        # Find client output directories (can have multiple for different services)
        client_dirs = list(output_base.glob(f"client_output_*_{run_id}"))
        if client_dirs:
            outputs["client_outputs"]["path"] = str(client_dirs[0])  # First for backward compat
            outputs["client_outputs"]["paths"] = [str(d) for d in client_dirs]
            outputs["client_outputs"]["exists"] = any(d.exists() for d in client_dirs)
            for client_dir in client_dirs:
                if client_dir.exists():
                    outputs["client_outputs"]["files"].extend([str(f) for f in client_dir.glob("*.jsonl")])
        
        # Find trace analysis directories (can have multiple for different services)
        trace_dirs = list(output_base.glob(f"trace_analysis_*_{run_id}"))
        if trace_dirs:
            outputs["trace_analysis"]["path"] = str(trace_dirs[0])  # First for backward compat
            outputs["trace_analysis"]["paths"] = [str(d) for d in trace_dirs]
            outputs["trace_analysis"]["exists"] = any(d.exists() for d in trace_dirs)
            for trace_dir in trace_dirs:
                if trace_dir.exists():
                    outputs["trace_analysis"]["files"].extend([str(f) for f in trace_dir.rglob("*")])
    else:
        # Use config to find exact paths
        if "dataset_dir" in config:
            dataset_path = Path(config["dataset_dir"])
            if run_id and not str(dataset_path).endswith(f"_{run_id}"):
                dataset_path = Path(str(dataset_path) + f"_{run_id}")
            outputs["dataset"]["path"] = str(dataset_path)
            outputs["dataset"]["exists"] = dataset_path.exists()
            if dataset_path.exists():
                outputs["dataset"]["files"] = [str(f) for f in dataset_path.glob("*.jsonl")]
        
        if "workload_dir" in config:
            workload_path = Path(config["workload_dir"])
            if run_id and not str(workload_path).endswith(f"_{run_id}"):
                workload_path = Path(str(workload_path) + f"_{run_id}")
            outputs["workload"]["path"] = str(workload_path)
            outputs["workload"]["exists"] = workload_path.exists()
            if workload_path.exists():
                outputs["workload"]["files"] = [str(f) for f in workload_path.glob("*.jsonl")]
        
        if "client_output" in config:
            client_path = Path(config["client_output"])
            if run_id and not str(client_path).endswith(f"_{run_id}"):
                client_path = Path(str(client_path) + f"_{run_id}")
            outputs["client_outputs"]["path"] = str(client_path)
            outputs["client_outputs"]["exists"] = client_path.exists()
            if client_path.exists():
                outputs["client_outputs"]["files"] = [str(f) for f in client_path.glob("*.jsonl")]
        
        if "trace_output" in config:
            trace_path = Path(config["trace_output"])
            if run_id and not str(trace_path).endswith(f"_{run_id}"):
                trace_path = Path(str(trace_path) + f"_{run_id}")
            outputs["trace_analysis"]["path"] = str(trace_path)
            outputs["trace_analysis"]["exists"] = trace_path.exists()
            if trace_path.exists():
                outputs["trace_analysis"]["files"] = [str(f) for f in trace_path.rglob("*")]
    
    return outputs

def copy_outputs(source_run_id, target_run_id, output_types=None, use_symlink=False):
    """
    Copy or link outputs from source run to target run.
    
    Args:
        source_run_id: Source run ID
        target_run_id: Target run ID
        output_types: List of output types to copy (dataset, workload, client_outputs, trace_analysis)
                     If None, copies all available
        use_symlink: If True, create symlinks instead of copying
    """
    if output_types is None:
        output_types = ["dataset", "workload", "client_outputs", "trace_analysis"]
    
    # Discover source outputs
    source_outputs = discover_run_outputs(source_run_id)
    
    # Discover target paths (need to get config for target run)
    # For now, we'll use the same pattern as source but with target_run_id
    target_outputs = {}
    
    copied = []
    failed = []
    
    for output_type in output_types:
        if output_type not in source_outputs:
            continue
        
        source_info = source_outputs[output_type]
        if not source_info["exists"]:
            print(f"  ⚠ {output_type}: Source output not found")
            failed.append(output_type)
            continue
        
        source_path = Path(source_info["path"])
        
        # Generate target path by replacing run_id in path
        target_path_str = str(source_path).replace(f"_{source_run_id}", f"_{target_run_id}")
        target_path = Path(target_path_str)
        
        try:
            if use_symlink:
                # Create symlink
                if target_path.exists():
                    if target_path.is_symlink():
                        target_path.unlink()
                    else:
                        shutil.rmtree(target_path)
                
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.symlink_to(source_path.absolute())
                print(f"  ✓ {output_type}: Linked {source_path.name} -> {target_path.name}")
            else:
                # Copy
                if target_path.exists():
                    if target_path.is_symlink():
                        target_path.unlink()
                    else:
                        shutil.rmtree(target_path)
                
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, target_path)
                print(f"  ✓ {output_type}: Copied {source_path.name} -> {target_path.name}")
            
            copied.append(output_type)
        except Exception as e:
            print(f"  ✗ {output_type}: Failed - {e}")
            failed.append(output_type)
    
    return copied, failed

def list_run_outputs(run_id):
    """List outputs for a specific run."""
    outputs = discover_run_outputs(run_id)
    
    print(f"\nOutputs for run: {run_id}")
    print("=" * 60)
    
    for output_type, info in outputs.items():
        status = "✓" if info["exists"] else "✗"
        print(f"\n{status} {output_type.replace('_', ' ').title()}:")
        
        # Show all paths if multiple exist
        if info.get("paths") and len(info["paths"]) > 1:
            print(f"    Paths ({len(info['paths'])}):")
            for path in info["paths"]:
                path_obj = Path(path)
                exists_marker = "✓" if path_obj.exists() else "✗"
                print(f"      {exists_marker} {path_obj.name}")
        elif info["path"]:
            print(f"    Path: {info['path']}")
            print(f"    Exists: {info['exists']}")
        
        if info["files"]:
            print(f"    Files: {len(info['files'])}")
            for file in info["files"][:5]:  # Show first 5
                print(f"      - {Path(file).name}")
            if len(info["files"]) > 5:
                print(f"      ... and {len(info['files']) - 5} more")
        elif not info["path"]:
            print(f"    Not found")
    
    print()

def list_all_run_outputs():
    """List outputs for all runs."""
    registry = load_runs_registry()
    
    if not registry:
        print("No runs found")
        return
    
    print("\n" + "=" * 60)
    print("Outputs for All Runs")
    print("=" * 60)
    
    for run_id in sorted(registry.keys(), reverse=True):
        run_info = registry[run_id]
        outputs = discover_run_outputs(run_id)
        
        # Count available outputs
        available = sum(1 for info in outputs.values() if info["exists"])
        total = len(outputs)
        
        print(f"\n{run_id} ({run_info.get('name', run_id)}):")
        print(f"  Outputs: {available}/{total} available")
        for output_type, info in outputs.items():
            if info["exists"]:
                # Show all paths if multiple
                if info.get("paths") and len(info["paths"]) > 1:
                    print(f"    ✓ {output_type} ({len(info['paths'])}):")
                    for path in info["paths"]:
                        print(f"      - {Path(path).name}")
                elif info["path"]:
                    print(f"    ✓ {output_type}: {Path(info['path']).name}")
    
    print()

def check_run_outputs(run_id):
    """Check if outputs exist for a run."""
    outputs = discover_run_outputs(run_id)
    
    available = sum(1 for info in outputs.values() if info["exists"])
    total = len(outputs)
    
    if available == 0:
        print(f"✗ No outputs found for run: {run_id}")
        return False
    elif available < total:
        print(f"⚠ Partial outputs ({available}/{total}) for run: {run_id}")
        return True
    else:
        print(f"✓ All outputs ({available}/{total}) available for run: {run_id}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Manage run outputs")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List outputs for a run
    list_parser = subparsers.add_parser("list", help="List outputs for a run")
    list_parser.add_argument("run_id", type=str, nargs="?", help="Run ID (default: current run)")
    
    # List outputs for all runs
    subparsers.add_parser("list-all", help="List outputs for all runs")
    
    # Copy outputs
    copy_parser = subparsers.add_parser("copy", help="Copy outputs from one run to another")
    copy_parser.add_argument("source_run_id", type=str, help="Source run ID")
    copy_parser.add_argument("target_run_id", type=str, help="Target run ID")
    copy_parser.add_argument("--dataset", action="store_true", help="Copy dataset")
    copy_parser.add_argument("--workload", action="store_true", help="Copy workload")
    copy_parser.add_argument("--client", action="store_true", help="Copy client outputs")
    copy_parser.add_argument("--trace", action="store_true", help="Copy trace analysis")
    
    # Link outputs
    link_parser = subparsers.add_parser("link", help="Link outputs from one run to another")
    link_parser.add_argument("source_run_id", type=str, help="Source run ID")
    link_parser.add_argument("target_run_id", type=str, help="Target run ID")
    link_parser.add_argument("--dataset", action="store_true", help="Link dataset")
    link_parser.add_argument("--workload", action="store_true", help="Link workload")
    link_parser.add_argument("--client", action="store_true", help="Link client outputs")
    link_parser.add_argument("--trace", action="store_true", help="Link trace analysis")
    
    # Check outputs
    check_parser = subparsers.add_parser("check", help="Check if outputs exist for a run")
    check_parser.add_argument("run_id", type=str, nargs="?", help="Run ID (default: current run)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        run_id = args.run_id or get_current_run_id()
        if not run_id:
            print("Error: No run ID specified and no current run set")
            sys.exit(1)
        list_run_outputs(run_id)
    
    elif args.command == "list-all":
        list_all_run_outputs()
    
    elif args.command == "copy":
        output_types = []
        if args.dataset:
            output_types.append("dataset")
        if args.workload:
            output_types.append("workload")
        if args.client:
            output_types.append("client_outputs")
        if args.trace:
            output_types.append("trace_analysis")
        
        print(f"\nCopying outputs from {args.source_run_id} to {args.target_run_id}...")
        copied, failed = copy_outputs(args.source_run_id, args.target_run_id, output_types, use_symlink=False)
        print(f"\n✓ Copied: {', '.join(copied) if copied else 'none'}")
        if failed:
            print(f"✗ Failed: {', '.join(failed)}")
    
    elif args.command == "link":
        output_types = []
        if args.dataset:
            output_types.append("dataset")
        if args.workload:
            output_types.append("workload")
        if args.client:
            output_types.append("client_outputs")
        if args.trace:
            output_types.append("trace_analysis")
        
        print(f"\nLinking outputs from {args.source_run_id} to {args.target_run_id}...")
        copied, failed = copy_outputs(args.source_run_id, args.target_run_id, output_types, use_symlink=True)
        print(f"\n✓ Linked: {', '.join(copied) if copied else 'none'}")
        if failed:
            print(f"✗ Failed: {', '.join(failed)}")
    
    elif args.command == "check":
        run_id = args.run_id or get_current_run_id()
        if not run_id:
            print("Error: No run ID specified and no current run set")
            sys.exit(1)
        check_run_outputs(run_id)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

