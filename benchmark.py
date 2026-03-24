"""
Dense Video Captioning - Benchmark Suite
========================================

Unified evaluation runner supporting three modes:
1. End-to-End (e2e): Runs the full pipeline (Scene Detection -> Captioning -> Merging) and evaluates against GT.
2. Offline: Evaluates pre-generated JSON results against GT (decouples inference from scoring).
3. Oracle: Uses Ground Truth timestamps to evaluate the captioning model's upper bound performance.

Usage:
    python benchmark.py e2e --model blip --limit 10
    python benchmark.py offline --model git
    python benchmark.py oracle --model qwen --limit 5
"""

import argparse
import sys
import os
from pathlib import Path
from colorama import Fore, init

# Ensure imports work regardless of execution directory
sys.path.insert(0, str(Path(__file__).parent))

from evaluation import e2e
from evaluation import offline
from evaluation import oracle
from src.config import Config

# Initialize colorama
init(autoreset=True)

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments with comprehensive help messages."""
    parser = argparse.ArgumentParser(
        description="Dense Video Captioning Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run end-to-end evaluation on the first 5 videos
  python benchmark.py e2e --model blip --limit 5
  
  # Evaluate offline results with a custom IoU threshold
  python benchmark.py offline --model git --threshold 0.4
  
  # Oracle evaluation (upper bound) on a specific dataset
  python benchmark.py oracle --model qwen --json data/ground_truth/val_1.json
        """
    )
    
    # Required Mode Selection
    parser.add_argument(
        "mode", 
        choices=["e2e", "offline", "oracle"], 
        help="Evaluation mode: 'e2e' (full pipeline), 'offline' (pre-generated), 'oracle' (GT timestamps)"
    )

    # Model Configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default=Config.CAPTION_MODEL_TYPE, 
        choices=["blip", "git", "qwen"], 
        help="Caption model architecture (default: %(default)s)"
    )
    
    # Evaluation Parameters
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of videos to evaluate (useful for quick testing)"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=Config.EVAL_IOU_THRESHOLD, 
        help="IoU threshold for matching predictions to GT (default: %(default)s)"
    )
    
    # Data Paths
    parser.add_argument(
        "--json", 
        type=str, 
        default=os.path.join(Config.DATA_DIR, 'ground_truth', 'val_1.json'), 
        help="Path to ground truth JSON file"
    )
    
    parser.add_argument(
        "--videos", 
        type=str, 
        default=Config.VIDEOS_DIR, 
        help="Path to videos directory"
    )
    
    return parser.parse_args()

def validate_arguments(args: argparse.Namespace) -> bool:
    """Validates that necessary files and directories exist before running."""
    errors = []
    
    if not os.path.exists(args.json):
        errors.append(f"Ground truth file not found: {args.json}")
    
    if args.mode in ["e2e", "oracle"] and not os.path.exists(args.videos):
        errors.append(f"Videos directory not found: {args.videos}")
    
    if args.mode == "offline":
        results_dir = os.path.join(Config.RESULTS_DIR, args.model)
        if not os.path.exists(results_dir):
            errors.append(f"Results directory for model '{args.model}' not found at: {results_dir}")
    
    if errors:
        print(Fore.RED + "❌ Validation Errors:")
        for error in errors:
            print(Fore.RED + f"   - {error}")
        return False
    
    return True

def print_header(args: argparse.Namespace) -> None:
    """Prints a summary of the benchmark configuration."""
    mode_descriptions = {
        "e2e": "End-to-End Inference (Full Pipeline)",
        "offline": "Offline Evaluation (Existing Results)",
        "oracle": "Oracle Evaluation (GT Timestamps)"
    }
    
    print(Fore.CYAN + "="*60)
    print(Fore.CYAN + "🚀 DENSE VIDEO CAPTIONING BENCHMARK")
    print(Fore.CYAN + "="*60)
    print(Fore.GREEN + f"📋 Mode:      {mode_descriptions[args.mode]}")
    print(Fore.GREEN + f"🔧 Model:     {args.model.upper()}")
    print(Fore.GREEN + f"📊 Threshold: IoU >= {args.threshold}")
    
    if args.limit:
        print(Fore.YELLOW + f"⚠️  Limit:     {args.limit} videos (Test Mode)")
    
    print(Fore.CYAN + "="*60 + "\n")

def main():
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    print_header(args)

    try:
        # Dispatch to the appropriate evaluation module
        if args.mode == "e2e":
            e2e.run(args)
        elif args.mode == "offline":
            offline.run(args)
        elif args.mode == "oracle":
            oracle.run(args)
        
        print(Fore.GREEN + "\n✅ Evaluation suite completed successfully!")
            
    except KeyboardInterrupt:
        print(Fore.RED + "\n❌ Evaluation aborted by user.")
        sys.exit(130)
        
    except Exception as e:
        print(Fore.RED + f"\n❌ Runtime Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()