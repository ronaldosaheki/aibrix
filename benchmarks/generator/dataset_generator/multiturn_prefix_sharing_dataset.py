import argparse
import json
import numpy as np
import logging
import os
from transformers import AutoTokenizer
from typing import Dict, Any
from generator.dataset_generator.synthetic_prompt import generate_synthetic_prompt
from generator.dataset_generator.util import save_dataset_jsonl

class NormalSampler:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std)
        
    def sample(self):
        sample = np.random.normal(self.mean, self.std)
        return sample
    
class ParetoSampler:
    def __init__(self, mean: float, std: float) -> None:
        mu = mean
        sigma = std
        r = sigma / mu
        alpha = 2 + 1 / r**2
        x_m = mu * (alpha - 1) / alpha
        self.samples = (np.random.pareto(alpha, size=100000) + 1) * x_m
        
    def sample(self):
        return self.samples[np.random.randint(0, len(self.samples))]
    
def analyze_dataset(dataset: Dict[str, Any], tokenizer: AutoTokenizer):
    token_lengths = []
    turns_counts = []
    for session in dataset:
        for prompt in session["prompts"]:
            text = prompt
            tokens = tokenizer.encode(text)
            token_lengths.append(len(tokens))
        turns_counts.append(len(session["prompts"]))
    analyze_dataset_from_stats(token_lengths, turns_counts, tokenizer)

def analyze_dataset_from_stats(token_lengths: list, turns_counts: list, tokenizer: AutoTokenizer):
    """Analyze dataset statistics from pre-computed token lengths and turn counts."""
    if len(token_lengths) == 0 or len(turns_counts) == 0:
        logging.warning("No data to analyze")
        return
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(token_lengths, percentiles)
    logging.warning(f"Token lengths statistics: mean {np.mean(token_lengths):.2f} std {np.std(token_lengths):.2f}")
    for p, v in zip(percentiles, values):
        logging.warning(f"  {p}th percentile: {v:.2f} tokens")
    values = np.percentile(turns_counts, percentiles)
    logging.warning(f"Turn count statistics: mean {np.mean(turns_counts):.2f} std {np.std(turns_counts):.2f}")
    for p, v in zip(percentiles, values):
        logging.warning(f"  {p}th percentile: {v:.2f} turns")
    
def main(args):
    session_sampler = NormalSampler(args.num_sessions_mean, args.num_sessions_std)
    turn_sampler = NormalSampler(args.num_turns_mean, args.num_turns_std)
    prompt_len_sampler = ParetoSampler(args.prompt_length_mean, args.prompt_length_std)
    
    num_sessions = int(session_sampler.sample())
    # Use hf_token from args if provided, otherwise fall back to HF_TOKEN env var
    hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    tokenizer_kwargs = {
        "legacy": True,
        "model_max_length": 4096,  # Increased to handle longer prefixes
        "padding_side": "right",
        "truncation_side": "right",
        "use_fast": True
    }
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        **tokenizer_kwargs
    )
    shared_prefix = ""
    if args.shared_prefix_len > 0:
        shared_prefix, _ = generate_synthetic_prompt(tokenizer, args.shared_prefix_len)
    
    # Open file for streaming write
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Keep minimal data in memory for analysis (just token lengths and turn counts)
    token_lengths = []
    turns_counts = []
    
    with open(args.output, 'w') as f:
        for session_id in range(0, num_sessions):
            num_turns = int(turn_sampler.sample())
            if num_turns <= 0:
                num_turns = 1
            flat_prompts_data = []
            session_token_lengths = []
            for i in range(0, num_turns):
                prompt_length = prompt_len_sampler.sample()
                if prompt_length <= 0:
                    print(f"sampled prompt length: {prompt_length}")
                prompt, token_count = generate_synthetic_prompt(tokenizer, int(prompt_length))
                # Process the prompt as needed
                if len(prompt) == 0:
                    print("Prompt is empty, skipping...")
                if i == 0:
                    prompt = shared_prefix + prompt
                flat_prompts_data.append(prompt)
                # Track token length for analysis
                tokens = tokenizer.encode(prompt)
                session_token_lengths.append(len(tokens))
            
            # Write session immediately to disk
            session_data = {
                "session_id": session_id,
                "prompts": flat_prompts_data,
            }
            f.write(json.dumps(session_data) + '\n')
            f.flush()  # Ensure data is written to disk immediately
            
            # Store minimal stats for analysis
            token_lengths.extend(session_token_lengths)
            turns_counts.append(num_turns)
            
            # Log progress every 500 sessions
            if (session_id + 1) % 500 == 0:
                logging.warning(f"Generated {session_id + 1}/{num_sessions} sessions...")
    
    # Analyze using token lengths and turn counts we tracked
    analyze_dataset_from_stats(token_lengths, turns_counts, tokenizer)
    logging.warning(f"...Finished saving dataset {args.output}.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure workload parameters.")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-llm-7b-chat", help="Name of the tokenizer.")
    parser.add_argument("--shared-prefix-len", type=int, default=0, help="Length of the shared prefix (simluating shared system prompt).")
    parser.add_argument("--prompt-length-mean", type=int, default=100, help="Length of the prompt (mean).")
    parser.add_argument("--prompt-length-std", type=int, default=10, help="Length of the prompt (std).")
    parser.add_argument("--num-turns-mean", type=float, default=10, help="Number of turns (mean).")
    parser.add_argument("--num-turns-std", type=float, default=1, help="Number of turns (std).")
    parser.add_argument("--num-sessions-mean", type=int, default=10, help="Number of sessions (mean).")
    parser.add_argument("--num-sessions-std", type=int, default=10, help="Number of sessions (std).")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for tokenizer authentication.")
    parser.add_argument("--output", type=str, default="output.jsonl", help="Output file name.")
    
    args = parser.parse_args()
    main(args)
    