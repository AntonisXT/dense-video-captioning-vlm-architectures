import argparse
import sys
import os
from pathlib import Path
from colorama import Fore, init

# Ensure imports work regardless of execution directory
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.visualizer import EvaluationVisualizer, compare_models
from src.config import Config

init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation Results Visualization Tool (Plot Generator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  1) Analyze one or more models (individual plots for each):
     python generate_plots.py analyze --models blip git --mode e2e

  2) Compare models (grouped bar chart):
     python generate_plots.py compare --models blip git qwen --mode all

  3) Analyze a specific file (custom path):
     python generate_plots.py file --path evaluation/reports/e2e/e2e_results_blip.json
"""
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Operation mode")

    # --- COMMAND 1: ANALYZE (Individual Plots) ---
    parser_analyze = subparsers.add_parser('analyze', help="Generate detailed plots for each model individually")
    parser_analyze.add_argument('--models', nargs='+', required=True, help="List of models (e.g., blip git)")
    parser_analyze.add_argument('--mode', type=str, default='e2e',
                               choices=['e2e', 'offline', 'oracle', 'all'],
                               help="Evaluation mode (default: e2e)")

    # --- COMMAND 2: COMPARE (Comparison Plot) ---
    parser_compare = subparsers.add_parser('compare', help="Generate a model comparison bar chart")
    parser_compare.add_argument('--models', nargs='+', required=True, help="List of models to compare")
    parser_compare.add_argument('--mode', type=str, default='e2e',
                               choices=['e2e', 'offline', 'oracle', 'all'],
                               help="Evaluation mode (default: e2e)")

    # --- COMMAND 3: FILE (Specific File) ---
    parser_file = subparsers.add_parser('file', help="Generate plots from a specific JSON file")
    parser_file.add_argument('--path', type=str, required=True, help="Path to the JSON results file")

    return parser.parse_args()


def _find_latest_report(base_dir: str, pattern: str):
    """Return newest file matching glob pattern inside base_dir."""
    from glob import glob
    candidates = glob(os.path.join(base_dir, pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def get_result_path(model, mode):
    """Resolves the result file path based on model name and mode.

    - e2e/oracle usually write stable filenames (e2e_results_{model}.json / oracle_results_{model}.json)
    - offline writes timestamped reports (offline_report_{model}_YYYYMMDD_HHMMSS.json), so we pick the newest.
    """
    if mode == 'e2e':
        base_dir = Config.EVAL_E2E_DIR
        stable = os.path.join(base_dir, f"e2e_results_{model}.json")
        return stable if os.path.exists(stable) else _find_latest_report(base_dir, f"e2e*{model}*.json")
    if mode == 'oracle':
        base_dir = Config.EVAL_ORACLE_DIR
        stable = os.path.join(base_dir, f"oracle_results_{model}.json")
        return stable if os.path.exists(stable) else _find_latest_report(base_dir, f"oracle*{model}*.json")
    if mode == 'offline':
        base_dir = Config.EVAL_OFFLINE_DIR
        return _find_latest_report(base_dir, f"offline_report_{model}_*.json")
    return None


def run_analysis(models, mode):
    modes = ['e2e', 'offline', 'oracle'] if mode == 'all' else [mode]
    print(Fore.CYAN + f"\n📊 Starting Analysis for: {', '.join(models)} (Mode: {mode.upper()})")

    for m in modes:
        print(Fore.CYAN + f"\n➡️  Mode: {m.upper()}")
        for model in models:
            path = get_result_path(model, m)
            if path and os.path.exists(path):
                print(Fore.YELLOW + f"\n   🔹 Processing: {model.upper()}...")
                try:
                    viz = EvaluationVisualizer(path)
                    viz.generate_all_plots()
                except Exception as e:
                    print(Fore.RED + f"   ❌ Error in {model}: {e}")
            else:
                print(Fore.RED + f"   ⚠️  File not found for {model} ({path})")


def run_comparison(models, mode):
    modes = ['e2e', 'offline', 'oracle'] if mode == 'all' else [mode]
    print(Fore.CYAN + f"\n⚖️  Model Comparison: {', '.join(models)} (Mode: {mode.upper()})")

    output_dir = os.path.join(Config.EVAL_PLOTS_DIR, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    for m in modes:
        file_paths = []
        for model in models:
            path = get_result_path(model, m)
            if path and os.path.exists(path):
                file_paths.append(path)
            else:
                print(Fore.RED + f"   ⚠️  Skipping {model} ({m}): File not found ({path})")

        if len(file_paths) < 2:
            print(Fore.RED + f"\n❌ Mode {m.upper()}: At least 2 valid result files are required for comparison.")
            continue

        compare_models(file_paths, output_dir)


def main():
    args = parse_args()
    os.makedirs(Config.EVAL_PLOTS_DIR, exist_ok=True)

    if args.command == 'analyze':
        run_analysis(args.models, args.mode)

    elif args.command == 'compare':
        run_comparison(args.models, args.mode)

    elif args.command == 'file':
        if os.path.exists(args.path):
            print(Fore.YELLOW + f"\n📂 Processing file: {args.path}")
            viz = EvaluationVisualizer(args.path)
            viz.generate_all_plots()
        else:
            print(Fore.RED + f"❌ File not found: {args.path}")


if __name__ == "__main__":
    main()
