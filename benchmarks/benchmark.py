import os
import sys
import subprocess
import yaml
import argparse
import logging 
from pathlib import Path
from client import (client, analyze)
from generator.dataset_generator import (synthetic_prefix_sharing_dataset, 
                                         multiturn_prefix_sharing_dataset, 
                                         utility)
from generator.workload_generator import workload_generator
from argparse import Namespace
from string import Template

# Helper function to print all override-able parameters
def print_override_help(config_path):
    with open(config_path, 'r') as f:
        content = os.path.expandvars(f.read())
        config = yaml.safe_load(content)

    print("\n========== OVERRIDE-ABLE PARAMETERS ==========")
    print("Use --override key=value to override these parameters.")
    print("Nested parameters can be accessed with dot notation (e.g., dataset_configs.synthetic_multiturn.shared_prefix_length)")
    print("\nTop-level parameters:")
    for key, value in config.items():
        if not isinstance(value, dict):
            print(f"  - {key}: {value}")
        else:
            print(f"\nNested parameters under '{key}':")
            print_nested_parameters(value, prefix=f"{key}.")
    print("\n=============================================")
    sys.exit(0)

def print_nested_parameters(config_dict, prefix=''):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print_nested_parameters(value, prefix=f"{prefix}{key}.")
        else:
            print(f"  - {prefix}{key}: {value}")


def deep_merge(base_dict, override_dict):
    """
    Deep merge two dictionaries, with override_dict taking precedence.
    Handles nested dictionaries recursively.
    """
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

class BenchmarkRunner:
    def __init__(self, config_base="config/base.yaml", overrides=None, service_config=None, run_id=None, reuse_from_run=None):
        self.overrides = overrides or []
        self.service_config = service_config
        self.run_id = run_id
        self.reuse_from_run = reuse_from_run
        self.reuse_config = {}  # Will store which outputs to reuse
        
        # Parse reuse_from_run if provided (format: "run-id:dataset,workload" or just "run-id")
        if self.reuse_from_run:
            self.parse_reuse_config()
        
        self.load_config(config_base)
        
        # Apply run_id to output paths if provided
        if self.run_id:
            self.apply_run_id_to_paths()
        
        # Apply reuse paths if specified
        if self.reuse_from_run:
            self.apply_reuse_paths()
        
        logging.warning(f"Loaded Config: {self.config}")
        for dir_key in ["dataset_dir", "workload_dir", "client_output", "trace_output"]:
            path_str = self.config.get(dir_key, "")
            if path_str:
                self.ensure_directories(path_str)
    
    def parse_reuse_config(self):
        """Parse reuse_from_run string to determine what to reuse."""
        # Format: "run-id" or "run-id:dataset,workload" or "run-id:all"
        if ":" in self.reuse_from_run:
            run_id, reuse_types = self.reuse_from_run.split(":", 1)
            if reuse_types == "all":
                self.reuse_config = {
                    "dataset": run_id,
                    "workload": run_id,
                    "client_outputs": run_id,
                    "trace_analysis": run_id
                }
            else:
                for reuse_type in reuse_types.split(","):
                    reuse_type = reuse_type.strip()
                    if reuse_type in ["dataset", "workload", "client_outputs", "trace_analysis"]:
                        self.reuse_config[reuse_type] = run_id
        else:
            # Default: reuse all if just run-id provided
            run_id = self.reuse_from_run
            self.reuse_config = {
                "dataset": run_id,
                "workload": run_id,
                "client_outputs": run_id,
                "trace_analysis": run_id
            }
    
    def apply_reuse_paths(self):
        """Apply paths from reused run to current config."""
        try:
            from run_outputs import discover_run_outputs
            
            for output_type, source_run_id in self.reuse_config.items():
                source_outputs = discover_run_outputs(source_run_id, config=self.config)
                
                if output_type == "dataset" and source_outputs["dataset"]["exists"]:
                    # Use source dataset path
                    self.config["dataset_dir"] = source_outputs["dataset"]["path"]
                    # Update dataset_file
                    if "dataset_file" in self.config:
                        dataset_file = Path(source_outputs["dataset"]["path"]) / Path(self.config["dataset_file"]).name
                        self.config["dataset_file"] = str(dataset_file)
                    logging.info(f"Reusing dataset from run: {source_run_id}")
                
                elif output_type == "workload" and source_outputs["workload"]["exists"]:
                    # Use source workload path
                    self.config["workload_dir"] = source_outputs["workload"]["path"]
                    # Update workload_file
                    if "workload_file" in self.config:
                        workload_file = Path(source_outputs["workload"]["path"]) / Path(self.config["workload_file"]).name
                        self.config["workload_file"] = str(workload_file)
                    logging.info(f"Reusing workload from run: {source_run_id}")
                
                elif output_type == "client_outputs" and source_outputs["client_outputs"]["exists"]:
                    # Use source client output path
                    self.config["client_output"] = source_outputs["client_outputs"]["path"]
                    logging.info(f"Reusing client outputs from run: {source_run_id}")
                
                elif output_type == "trace_analysis" and source_outputs["trace_analysis"]["exists"]:
                    # Use source trace analysis path
                    self.config["trace_output"] = source_outputs["trace_analysis"]["path"]
                    logging.info(f"Reusing trace analysis from run: {source_run_id}")
        except ImportError:
            logging.warning("run_outputs module not available, cannot reuse outputs")
    
    def apply_run_id_to_paths(self):
        """Apply run_id to output paths."""
        if not self.run_id:
            return
        
        # Paths to modify with run_id
        path_keys = ["dataset_dir", "workload_dir", "client_output", "trace_output"]
        
        for key in path_keys:
            if key in self.config and self.config[key]:
                original_path = self.config[key]
                # Add run_id to path
                path = Path(original_path)
                # Append run_id to the directory name
                if path.name:
                    # For paths like "./output/dataset_gpt-oss-120" -> "./output/dataset_gpt-oss-120_run-id"
                    self.config[key] = str(path.parent / f"{path.name}_{self.run_id}")
                else:
                    # For paths ending with /, append run_id
                    self.config[key] = f"{original_path.rstrip('/')}_{self.run_id}"
        
        # Update workload_file to match new workload_dir
        if "workload_file" in self.config and "workload_dir" in self.config:
            workload_file = self.config["workload_file"]
            workload_dir = self.config["workload_dir"]
            # Replace template variable or update path
            if "${workload_dir}" in workload_file:
                self.config["workload_file"] = workload_file.replace("${workload_dir}", workload_dir)
            else:
                # Update path to match new workload_dir
                old_workload_dir = Path(workload_file).parent
                new_workload_file = Path(workload_dir) / Path(workload_file).name
                self.config["workload_file"] = str(new_workload_file)
        
        # Update dataset_file to match new dataset_dir
        if "dataset_file" in self.config and "dataset_dir" in self.config:
            dataset_file = self.config["dataset_file"]
            dataset_dir = self.config["dataset_dir"]
            if "${dataset_dir}" in dataset_file:
                self.config["dataset_file"] = dataset_file.replace("${dataset_dir}", dataset_dir)
            else:
                # Update path to match new dataset_dir
                new_dataset_file = Path(dataset_dir) / Path(dataset_file).name
                self.config["dataset_file"] = str(new_dataset_file)
            
    def apply_overrides(self, config_dict):
        for override in self.overrides:
            if '=' not in override:
                logging.warning(f"Invalid override format: {override}. Use key=value.")
                continue
            key, value = override.split("=", 1)
            try:
                parsed_value = yaml.safe_load(value)
            except Exception:
                parsed_value = value
            # Handle nested keys like "workload_configs.target_qps"
            parts = key.split(".")
            d = config_dict
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = parsed_value
            logging.info(f"Overridden {key} = {parsed_value}")
        return config_dict

    def load_config(self, config_path):
        if not Path(config_path).is_file():
            logging.error(f"{config_path} not found.")
            sys.exit(1)

        logging.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            content = os.path.expandvars(f.read())
            raw_config = yaml.safe_load(content) or {}
        
        # If service_config is provided, merge it with base config
        if self.service_config:
            logging.info(f"Merging service-specific overrides")
            raw_config = deep_merge(raw_config, self.service_config)
        
        # Apply command-line overrides
        raw_config = self.apply_overrides(raw_config)
        
        # Resolve template variables
        resolved_config = {}
        for key, value in raw_config.items():
            if isinstance(value, str):
                template = Template(value)
                value = template.safe_substitute(raw_config)
            resolved_config[key] = value
        self.config = resolved_config

    def ensure_directories(self, path_str):
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)

    def generate_dataset(self):
        # Check if reusing dataset
        if "dataset" in self.reuse_config:
            dataset_file = Path(self.config["dataset_file"])
            if dataset_file.exists():
                logging.info(f"Skipping dataset generation - reusing from run: {self.reuse_config['dataset']}")
                logging.info(f"Using dataset: {dataset_file}")
                return
            else:
                logging.warning(f"Reuse specified but dataset not found: {dataset_file}")
        
        dataset_type = self.config["prompt_type"]
        logging.info(f"Generating synthetic dataset {dataset_type}...")

        if dataset_type not in self.config["dataset_configs"]:
            logging.error(f"Unknown prompt type: {dataset_type}")
            sys.exit(1)

        subconfig = self.config["dataset_configs"][dataset_type]
        dataset_file = self.config["dataset_file"]  # Use the pre-defined dataset_file

        if dataset_type == "synthetic_shared":
            args_dict = {
                "output": dataset_file,
                "randomize_order": True,
                "tokenizer": self.config["tokenizer"],
                "app_name": self.config["prompt_type"],
                "prompt_length": subconfig["prompt_length"],
                "prompt_length_std": subconfig["prompt_std"],
                "shared_proportion": subconfig["shared_prop"],
                "shared_proportion_std": subconfig["shared_prop_std"],
                "num_samples_per_prefix": subconfig["num_samples"],
                "num_prefix": subconfig["num_prefix"],
                "num_configs": subconfig.get("num_dataset_configs", 1),
                "to_workload": False,
                "rps": 0,
            }
            args = Namespace(**args_dict)
            synthetic_prefix_sharing_dataset.main(args)
            
        elif dataset_type == "synthetic_multiturn":
            # Get hf_token from config or environment for Hugging Face authentication
            hf_token = self.config.get('hf_token') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            args_dict = {
                "output": dataset_file,
                "tokenizer": self.config["tokenizer"],
                "shared_prefix_len": subconfig["shared_prefix_length"],
                "prompt_length_mean": subconfig["prompt_length"],
                "prompt_length_std": subconfig["prompt_std"],
                "num_turns_mean": subconfig["num_turns"],
                "num_turns_std": subconfig["num_turns_std"],
                "num_sessions_mean": subconfig["num_sessions"],
                "num_sessions_std": subconfig["num_sessions_std"],
            }
            if hf_token:
                args_dict["hf_token"] = hf_token
            args = Namespace(**args_dict)
            multiturn_prefix_sharing_dataset.main(args)

        elif dataset_type == "client_trace":
            args_dict = {
                "command": "convert",
                "path": subconfig["trace"],
                "type": "trace",
                "output": dataset_file,
            }
            args = Namespace(**args_dict)
            utility.main(args)
            
        elif dataset_type == "sharegpt":
            if not Path(subconfig["target_dataset"]).is_file():
                print("[INFO] Downloading ShareGPT dataset...")
                subprocess.run([
                    "wget", "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                    "-O", subconfig["target_dataset"]
                ], check=True)
            args_dict = {
                "command": "convert",
                "path": subconfig["target_dataset"],
                "type": "sharegpt",
                "output": dataset_file,
            }
            args = Namespace(**args_dict)
            utility.main(args)

    def generate_workload(self):
        # Check if reusing workload
        if "workload" in self.reuse_config:
            workload_file = Path(self.config["workload_file"])
            if workload_file.exists():
                logging.info(f"Skipping workload generation - reusing from run: {self.reuse_config['workload']}")
                logging.info(f"Using workload: {workload_file}")
                return
            else:
                logging.warning(f"Reuse specified but workload not found: {workload_file}")
        
        workload_type = self.config["workload_type"]
        print("[INFO] Generating workload...")
        
        if workload_type not in self.config["workload_configs"]:
            print(f"[ERROR] Unknown workload type: {workload_type}")
            sys.exit(1)

        subconfig = self.config["workload_configs"][workload_type]
        workload_type_dir = self.config["workload_dir"]
        self.ensure_directories(workload_type_dir)
        
        dataset_file = self.config["dataset_file"]  # Use the pre-defined dataset_file
        args_dict = {
            "prompt_file": dataset_file,
            "interval_ms": self.config["interval_ms"],
            "duration_ms": self.config["duration_ms"],
            "trace_type": workload_type,
            "tokenizer": self.config["tokenizer"],
            "output_dir": workload_type_dir,
            "output_format": "jsonl",
        }

        if workload_type == "constant":
            args_dict.update({
                "target_qps": subconfig["target_qps"],
                "target_prompt_len": subconfig["target_prompt_len"],
                "target_completion_len": subconfig["target_completion_len"],
                "max_concurrent_sessions": subconfig.get("max_concurrent_sessions", 1),
            })
            
        elif workload_type == "synthetic":
            if subconfig["use_preset_pattern"]:
                patterns = subconfig["preset_patterns"]
                pattern_args = {
                    "traffic_pattern": patterns["traffic_pattern"],
                    "prompt_len_pattern": patterns["prompt_len_pattern"],
                    "completion_len_pattern": patterns["completion_len_pattern"],
                }
            else:
                pattern_files = subconfig["pattern_files"]
                pattern_args = {
                    "traffic_pattern_config": pattern_files["traffic_file"],
                    "prompt_len_pattern_config": pattern_files["prompt_len_file"],
                    "completion_len_pattern_config": pattern_files["completion_len_file"],
                }
            pattern_args["max_concurrent_sessions"] = subconfig["pattern_files"].get("max_concurrent_sessions", 1)
            args_dict.update(pattern_args)
            
        elif workload_type == "stat":
            args_dict.update({
                "stat_trace_type": subconfig["stat_trace_type"],
                "traffic_file": subconfig["traffic_file"],
                "prompt_len_file": subconfig["prompt_len_file"],
                "completion_len_file": subconfig["completion_len_file"],
                "qps_scale": subconfig["qps_scale"],
                "output_scale": subconfig["output_scale"],
                "input_scale": subconfig["input_scale"],
            })
            
        elif workload_type == "azure":
            if not Path(subconfig["trace_path"]).is_file():
                logging.info("Downloading Azure dataset...")
                subprocess.run([
                    "wget", "https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_conv.csv",
                    "-O", subconfig["trace_path"]
                ], check=True)
            args_dict.update({
                "traffic_file": subconfig["trace_path"],
                "group_interval_seconds": 1,
            })
            
        elif workload_type == "mooncake":
            if not Path(subconfig["trace_path"]).is_file():
                logging.info("Downloading Mooncake dataset...")
                if subconfig["trace_type"] == "conversation":
                    subprocess.run([
                        "wget", "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/conversation_trace.jsonl",
                        "-O", subconfig["trace_path"]
                    ], check=True)
                elif subconfig["trace_type"] == "synthetic":
                    subprocess.run([
                        "wget", "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/synthetic_trace.jsonl",
                        "-O", subconfig["trace_path"]
                    ], check=True)
                elif subconfig["trace_type"] == "toolagent":
                    subprocess.run([
                        "wget", "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl",
                        "-O", subconfig["trace_path"]
                    ], check=True)
                else:
                    trace_type = subconfig["trace_type"]
                    logging.error(f"Unknown trace type: {trace_type}")
                    logging.error("Choose among [conversation|synthetic|toolagent]")
                    sys.exit(1)
            args_dict.update({
                "traffic_file": subconfig["trace_path"],
                "prompt_file": None,
            })
                    

        args = Namespace(**args_dict)
        logging.info(f"Running workload generator with args: {args}")
        workload_generator.main(args)

    def run_client(self):
        logging.info("Running client to dispatch workload...")
        workload_file = self.config["workload_file"]  # Use the pre-defined workload_file
        # Only add api_key if it's not None
        # Special handling for API_KEY
        if 'api_key' in self.config and self.config["api_key"] == '${API_KEY}':
            # API_KEY was not set in environment variables
            logging.warning('No API_KEY provided.')
            # Set to None so it can be handled appropriately later
            self.config["api_key"] = None
        
        # Get client_duration_limit (in seconds)
        duration_limit = self.config.get("client_duration_limit", None)
        
        # Get client_max_concurrent_sessions from client config (runtime enforcement)
        # This is separate from workload generator's max_concurrent_sessions
        max_concurrent_sessions = self.config.get("client_max_concurrent_sessions", None)
        
        # Get client_max_requests from client config (limit total requests processed)
        max_requests = self.config.get("client_max_requests", None)
        
        # Get provider configuration
        provider = self.config.get("provider", "custom")
        openrouter_provider_config = self.config.get("openrouter_provider_config", None)
        
        # Convert openrouter_provider_config to JSON string if it exists
        openrouter_provider_config_str = None
        if openrouter_provider_config:
            import json
            openrouter_provider_config_str = json.dumps(openrouter_provider_config)
        
        args_dict = {
            "workload_path": workload_file,
            "endpoint": self.config["endpoint"],
            "model": self.config["target_model"],
            "api_key": self.config["api_key"],
            "output_file_path": f"{self.config['client_output']}/output.jsonl",
            "streaming": self.config.get("streaming_enabled", False),
            "routing_strategy": self.config.get("routing_strategy", "random"),
            "output_token_limit": self.config.get("output_token_limit", 128),
            "time_scale": self.config.get("time_scale", 1.0),
            "timeout_second": self.config.get("timeout_second", 60.0),
            "max_retries": self.config.get("max_retries", 0),
            "duration_limit": duration_limit,
            "max_concurrent_sessions": max_concurrent_sessions,
            "max_requests": max_requests,
            "provider": provider,
            "openrouter_provider_config": openrouter_provider_config_str,
        }
        args = Namespace(**args_dict)
        logging.info(f"Running client with args: {args}")
        client.main(args)

    def run_analysis(self):
        # Check if reusing trace analysis
        if "trace_analysis" in self.reuse_config:
            trace_output = Path(self.config["trace_output"])
            if trace_output.exists() and any(trace_output.iterdir()):
                logging.info(f"Skipping trace analysis - reusing from run: {self.reuse_config['trace_analysis']}")
                logging.info(f"Using trace analysis: {trace_output}")
                return
            else:
                logging.warning(f"Reuse specified but trace analysis not found: {trace_output}")
        
        logging.info("Analyzing trace output...")
        args_dict = {
            "trace": f"{self.config['client_output']}/output.jsonl",
            "output": self.config["trace_output"],
            "goodput_target": self.config["goodput_target"],
        }
        args = Namespace(**args_dict)
        logging.info(f"Running analysis with args: {args}")
        analyze.main(args)

    def run(self, command):
        logging.info("========== Starting Benchmark ==========")
        actions = {
            "dataset": self.generate_dataset,
            "workload": self.generate_workload,
            "client": self.run_client,
            "analysis": self.run_analysis,
            "all": lambda: [self.generate_dataset(), self.generate_workload(), self.run_client(), self.run_analysis()],
            "": lambda: [self.generate_dataset(), self.generate_workload(), self.run_client(), self.run_analysis()]
        }
        if command not in actions:
            logging.error(f"Unknown command: {command}")
            logging.error("Usage: script.py [dataset|workload|client|analysis|all]")
            sys.exit(1)
        result = actions[command]
        if callable(result):
            result()
        logging.info("========== Benchmark Completed ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark pipeline", add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--stage", help="Specify the benchmark stage to run. Possible stages:\n- all: Run all stages (dataset, workload, client, analysis)\n- dataset: Generate the dataset\n- workload: Generate the workload\n- client: Run the client to dispatch workload\n- analysis: Analyze the trace output")
    parser.add_argument("--config", required=True, help="Path to base config YAML")
    parser.add_argument("--override", action="append", default=[], help="Override config values in the config file specified through --config, e.g., --override time_scale=2.0 or target_qps=5. Use 'help' to list all override-able parameters.")
    

    args = parser.parse_args()

    # Check if user asked for help on overrides
    if 'help' in args.override:
        print_override_help(args.config)
    elif not args.stage:
        parser.error("--stage is required")

    runner = BenchmarkRunner(config_base=args.config, overrides=args.override)
    runner.run(args.stage)
