#!/usr/bin/env python
"""
Example script to run main_leaves.py with custom configuration.
This serves as a template for users to modify based on their needs.
"""

import os
from config import Config

# Import the main function
import sys
sys.path.insert(0, os.path.dirname(__file__))

# You can import main directly if needed
# from main_leaves import main

def run_example():
    """
    Example configuration for running tree search decoding.
    Modify these parameters based on your setup.
    """
    
    # Basic configuration
    config = Config(
        # Model selection
        model = "mist-7b",  # Options: mist-7b, gemma-2-2b, llama2-7b, llama-3.1-8B, etc.
        
        # Dataset selection
        dataset = "gsm8k",  # Options: gsm8k, multiarith, svamp, addsub, singleeq, etc.
        
        # Device configuration
        device = "cuda:0",  # Use "cuda:0", "cuda:1", etc. or "cpu"
        
        # Tree search parameters
        NODE_LENGTH = [1]*1 + [30]*9,  # Length of each node at different tree levels
        min_step = 6,                   # Minimum steps before collecting leaves
        batch_size = 3,                 # Batch size for parallel generation
        
        # Methods to run - can be multiple!
        methods = ["search"],  # Options: "search", "cot_decoding", "greedy", "zs_cot_prompt"
        
        # Path configuration - IMPORTANT: Update these paths!
        root_path="/hdd/zijianwang/CoT-decoding",  # Path containing dataset/ and Probe_weights/
        MODEL_CACHE_DIR="/hdd/zijianwang/HF_CACHE",  # HuggingFace model cache
        
        # Prober configuration
        prober_type = "LR_ps",  # Options: "SVM", "LR", "SVM_ps", "LR_ps"
        rep = "hiddens",        # Options: "hiddens", "mlp", "post-attn"
        score_layers = [30, 29, 28],  # Which transformer layers to use for scoring
    )
    
    # Additional configuration for output directories
    config.LEAVES_DIR = os.path.join(config.root_path, "Leaves_new")
    config.LOG_DIR = os.path.join(config.root_path, "Log_leaves")
    
    # Create necessary directories
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.LEAVES_DIR, exist_ok=True)
    
    # Print configuration summary
    print("=" * 80)
    print("Running main_leaves.py with the following configuration:")
    print("=" * 80)
    print(f"Model: {config.model} ({config.model_name})")
    print(f"Dataset: {config.dataset}")
    print(f"Device: {config.device}")
    print(f"Methods: {config.methods}")
    print(f"Prober: {config.prober_type} using {config.rep} representation")
    print(f"Score layers: {config.score_layers}")
    print(f"Output directories:")
    print(f"  - Logs: {config.LOG_DIR}")
    print(f"  - Leaves: {config.LEAVES_DIR}")
    print("=" * 80)
    
    # Uncomment the following line to actually run the main function
    # from main_leaves import main
    # main(config)
    
    print("\nConfiguration created successfully!")
    print("To run the actual experiment, uncomment the main() call in this script.")
    print("\nNote: Make sure you have:")
    print("  1. Dataset files at: {}/dataset/".format(config.root_path))
    print("  2. Prober weights at: {}/Probe_weights/{}/{}/{}/".format(
        config.root_path, config.model, config.dataset, config.prober_type))
    print("  3. Sufficient GPU memory for model: {}".format(config.model_name))

if __name__ == "__main__":
    run_example()

