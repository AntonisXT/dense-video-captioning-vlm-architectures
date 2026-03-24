"""
Visualization Module for Dense Video Captioning Evaluation
===========================================================
"""

import json
import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# Professional aesthetics using seaborn if available
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


class EvaluationVisualizer:
    """
    Generates 4 plots:
    1. Performance Summary (Bar Chart) - Overall metrics
    2. Temporal Localization Quality (Line Plot) - IoU analysis [e2e only]
    3. Score Distribution (Box Plot) - Quality consistency
    4. Caption Length Distribution (KDE Plot) - Linguistic verbosity
    """
    def __init__(self, results_path: str, output_dir: Optional[str] = None):
        self.results_path = results_path
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # Determine output directory
        if output_dir is None:
            base_dir = Path(results_path).parent.parent / "plots"
            self.output_dir = str(base_dir)
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = self.results.get("model", "unknown")
        self.mode = self.results.get("mode", "unknown")
        
        self.colors = {
            'precision': '#2E86AB',  
            'recall': '#A23B72',     
            'bleu3': '#F18F01',   
            'bleu4': '#D35400',     
            'meteor': '#C73E1D',   
            'rouge': '#06A77D',   
            'gen': '#E63946',       
            'gt': '#457B9D'         
        }

    def _save_figure(self, fig: plt.Figure, name: str) -> str:
        """Save figure with consistent naming and high quality."""
        filename = f"{self.mode}_{self.model}_{name}.png"
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 Saved: {filename}")
        plt.close(fig)
        return filepath

    def plot_performance_summary(self) -> Optional[str]:
        """
        PLOT 1: Main Performance Metrics   
        Clean bar chart showing all key metrics with clear labels.
        """
        summary = self.results.get("summary", self.results.get("metrics", {}))
        if not summary: 
            return None
        
        # Define metrics to display
        metrics_to_plot = []
        if self.mode != 'oracle':
            metrics_to_plot.extend([
                ('Precision', 'precision', self.colors['precision']),
                ('Recall', 'recall', self.colors['recall'])
            ])
        
        metrics_to_plot.extend([
            ('BLEU-3', 'BLEU_3', self.colors['bleu3']),
            ('BLEU-4', 'BLEU_4', self.colors['bleu4']),
            ('METEOR', 'METEOR', self.colors['meteor']),
            ('ROUGE-L', 'ROUGE_L', self.colors['rouge'])
        ])

        # Extract data
        labels, values, colors = [], [], []
        for label, key, color in metrics_to_plot:
            if key in summary:
                labels.append(label)
                values.append(summary[key])
                colors.append(color)

        if not values: 
            return None

        # Create figure with proper sizing
        fig, ax = plt.subplots(figsize=(11, 6))
        
        # Bar chart with enhanced styling
        bars = ax.bar(labels, values, color=colors, alpha=0.85, 
                     width=0.65, edgecolor='black', linewidth=1.5)
        
        # Styling
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)", fontsize=14, fontweight='bold')
        ax.set_title(f"Performance Metrics - {self.model.upper()}", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.2f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return self._save_figure(fig, "performance_summary")

    def plot_temporal_localization(self) -> Optional[str]:
        """
        PLOT 2: Temporal Localization Quality (IoU Analysis)
        Shows how well the system localizes events in time.
        Only for e2e/offline modes.
        """
        if self.mode == 'oracle': 
            return None
        
        summary = self.results.get("summary", {})
        curve = summary.get("temporal_curve", None)

        # Temporal curve must be pre-computed in e2e.py/offline.py
        if not curve or not isinstance(curve, list) or len(curve) < 3:
            print(f"  ⚠️  Skipping temporal localization plot: No valid curve data found")
            return None
        
        thresholds = np.array([c.get("iou", 0) for c in curve], dtype=float)
        precisions = np.array([c.get("precision", 0) for c in curve], dtype=float)
        recalls = np.array([c.get("recall", 0) for c in curve], dtype=float)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Line plot with markers
        ax.plot(thresholds, precisions, marker='o', markersize=8,
               linewidth=3, color=self.colors['precision'],
               label='Precision')
        
        if recalls is not None and len(recalls) > 0:
            ax.plot(thresholds, recalls, marker='s', markersize=7,
                   linewidth=2.5, color=self.colors['recall'],
                   label='Recall')
        
        # Fill area under curve
        ax.fill_between(thresholds, precisions, alpha=0.2, 
                       color=self.colors['precision'])
        
        # Styling
        ax.set_xlabel("IoU Threshold", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score (%)", fontsize=14, fontweight='bold')
        ax.set_title("Temporal Localization Quality", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Reference line
        ax.axvline(x=0.3, color='red', linestyle='--', linewidth=2, 
                  alpha=0.5, label='Standard IoU=0.3')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Limits and legend
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return self._save_figure(fig, "temporal_localization")

    def plot_score_distribution(self) -> Optional[str]:
        """
        PLOT 3: Score Distribution (Box Plot)
        Updated for BLEU-3 and BLEU-4
        """
        details = self.results.get("details", {})
        
        metric_keys = ['BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L']
        metric_data = {k: [] for k in metric_keys}
        
        for vid_data in details.values():
            if self.mode == 'oracle':
                scores_list = vid_data if isinstance(vid_data, list) else []
                for entry in scores_list:
                    s = entry.get('scores', {})
                    for k in metric_keys:
                        metric_data[k].append(s.get(k, 0) * 100)
            else:
                scores_list = vid_data.get("matched_scores", [])
                for s in scores_list:
                    for k in metric_keys:
                        metric_data[k].append(s.get(k, 0) * 100)
        
        if not any(metric_data.values()) or all(len(v) < 3 for v in metric_data.values()):
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 6))
        
        # Box plot
        bp = ax.boxplot(
            [metric_data[k] for k in metric_keys],
            labels=['BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L'],
            patch_artist=True,
            showmeans=True,
            meanline=False,
            widths=0.6,
            boxprops=dict(facecolor='lightsteelblue', alpha=0.7, linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(color='darkblue', linewidth=2.5),
            meanprops=dict(marker='D', markerfacecolor='red', 
                          markeredgecolor='darkred', markersize=8)
        )
        
        # Styling
        ax.set_ylabel("Score (%)", fontsize=14, fontweight='bold')
        ax.set_title(f"Caption Quality Distribution - {self.model.upper()}", 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 105)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color='darkblue', linewidth=2.5, label='Median'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
                  markersize=8, label='Mean'),
            Patch(facecolor='lightsteelblue', alpha=0.7, label='IQR (25-75%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', 
                 fontsize=10, framealpha=0.95)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return self._save_figure(fig, "score_distribution")

    def plot_caption_length_distribution(self) -> Optional[str]:
        """
        PLOT 4: Caption Length Distribution
        """
        details = self.results.get("details", {})
        gen_lens, gt_lens = [], []
        
        # Extract lengths
        for vid_data in details.values():
            if self.mode == 'oracle':
                entries = vid_data if isinstance(vid_data, list) else []
            else:
                entries = vid_data.get("logs", [])
            
            for entry in entries:
                # Generated caption length
                status = entry.get('status', '') if isinstance(entry, dict) else ''
                gen_text = entry.get('pred', entry.get('generated', ''))
                if isinstance(gen_text, str):
                    gen_norm = gen_text.strip()
                else:
                    gen_norm = ''
                # Exclude placeholders and unmatched GT rows (FALSE NEGATIVE has pred='N/A')
                if gen_norm and gen_norm.upper() != 'N/A' and status != 'FALSE NEGATIVE':
                    gen_lens.append(len(gen_norm.split()))
                
                # Ground truth length
                gt_text = entry.get("gt", entry.get("ground_truth", ""))
                if gt_text and gt_text != "N/A":
                    if isinstance(gt_text, list):
                        gt_text = gt_text[0]
                    if gt_text.strip():
                        gt_lens.append(len(gt_text.split()))
        
        if len(gen_lens) < 5:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 6))
        
        gen_mean = np.mean(gen_lens)
        gen_median = np.median(gen_lens)
        gt_mean = np.mean(gt_lens) if gt_lens else 0
        
        # Plot distributions
        try:
            if HAS_SEABORN and len(gen_lens) > 1:
                if len(gt_lens) > 1:
                    sns.kdeplot(gt_lens, color=self.colors['gt'], 
                               label=f'Ground Truth (μ={gt_mean:.1f})', 
                               fill=True, alpha=0.3, linewidth=2.5, 
                               ax=ax, warn_singular=False)
                
                sns.kdeplot(gen_lens, color=self.colors['gen'], 
                           label=f'{self.model.upper()} (μ={gen_mean:.1f})', 
                           fill=True, alpha=0.4, linewidth=2.5, 
                           ax=ax, warn_singular=False)
            else:
                if gt_lens:
                    ax.hist(gt_lens, bins=20, alpha=0.4, 
                           color=self.colors['gt'], label='Ground Truth', 
                           density=True, edgecolor='black')
                ax.hist(gen_lens, bins=20, alpha=0.5, 
                       color=self.colors['gen'], label=f'{self.model.upper()}', 
                       density=True, edgecolor='black')
        except Exception:
            if gt_lens:
                ax.hist(gt_lens, bins=15, alpha=0.4, color=self.colors['gt'], 
                       label='Ground Truth', density=True)
            ax.hist(gen_lens, bins=15, alpha=0.5, color=self.colors['gen'], 
                   label=f'{self.model.upper()}', density=True)
        
        # Add mean/median lines
        ax.axvline(gen_mean, color=self.colors['gen'], linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'Model Mean ({gen_mean:.1f})')
        
        if gt_lens and len(gt_lens) > 1:
            ax.axvline(gt_mean, color=self.colors['gt'], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'GT Mean ({gt_mean:.1f})')
        
        # Styling
        ax.set_xlabel("Caption Length (words)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Density", fontsize=14, fontweight='bold')
        ax.set_title(f"Caption Length Distribution - {self.model.upper()}", 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        
        max_len = max(max(gen_lens), max(gt_lens) if gt_lens else 0)
        ax.set_xlim(0, min(max_len + 5, 50))
        
        stats_text = (
            f'Model Statistics:\n'
            f'Mean: {gen_mean:.1f} words\n'
            f'Median: {gen_median:.1f} words\n'
            f'Range: {min(gen_lens)}-{max(gen_lens)}'
        )
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return self._save_figure(fig, "caption_length_distribution")

    def generate_all_plots(self):
        """Generate all essential plots."""
        print(f"\n📊 Generating Plots for: {self.model.upper()}")
        
        created = []
        result = self.plot_performance_summary()
        if result: created.append("Performance Summary")
        
        result = self.plot_temporal_localization()
        if result: created.append("Temporal Localization")
        
        result = self.plot_score_distribution()
        if result: created.append("Score Distribution")
        
        result = self.plot_caption_length_distribution()
        if result: created.append("Caption Length Distribution")
        
        print(f"✅ Created {len(created)} plots: {', '.join(created)}")


def _compare_models_single_mode(models_data: List[dict], mode: str, output_dir: str):
    """Generate a grouped bar chart for a single evaluation mode."""

    if mode == 'oracle':
        metrics = ['BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L']
        labels = ['BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L']
        title = 'Model Performance Comparison (Oracle / Caption Quality)'
    else:
        metrics = ['precision', 'recall', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L']
        labels = ['Precision', 'Recall', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L']
        title = f"Model Performance Comparison ({mode.upper()})"

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(metrics))
    n_models = len(models_data)
    width = 0.8 / max(n_models, 1)

    model_colors = ['#2E86AB', '#E63946', '#06A77D', '#F18F01', '#A23B72']

    for i, md in enumerate(models_data):
        values = [float(md['summary'].get(m, 0) or 0) for m in metrics]
        offset = (i - n_models/2 + 0.5) * width

        bars = ax.bar(
            x + offset, values, width,
            label=md['model'],
            color=model_colors[i % len(model_colors)],
            alpha=0.85, edgecolor='black', linewidth=1
        )

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold'
                )

    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, title='Models', title_fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"model_comparison_{mode}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved comparison: {output_path}")


def _extract_caption_lengths_from_report(report: dict):
    """Return (pred_lengths, gt_lengths) from a single report JSON dict."""
    pred_lens = []
    gt_lens = []
    details = report.get("details", {}) or {}

    if isinstance(details, list):
        video_items = enumerate(details)
    else:
        video_items = details.items()

    for _, vid_data in video_items:
        if isinstance(vid_data, list):
            entries = vid_data
        elif isinstance(vid_data, dict):
            entries = vid_data.get("logs") or vid_data.get("entries") or []
        else:
            entries = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            status = entry.get('status', '')
            gen = entry.get('pred') or entry.get('generated') or entry.get('caption') or ''
            if isinstance(gen, str):
                gen_norm = gen.strip()
            else:
                gen_norm = ''
            if gen_norm and gen_norm.upper() != 'N/A' and status != 'FALSE NEGATIVE':
                pred_lens.append(len(gen_norm.split()))

            gt = entry.get("gt") or entry.get("ground_truth") or ""
            if isinstance(gt, list) and gt:
                gt_txt = gt[0]
            else:
                gt_txt = gt

            if isinstance(gt_txt, str) and gt_txt.strip() and gt_txt.strip().upper() != "N/A":
                gt_lens.append(len(gt_txt.split()))

    return pred_lens, gt_lens


def _compare_caption_lengths_single_mode(models_data: List[dict], mode: str, output_dir: str):
    """COMPARISON PLOT: Caption Length Distribution (per mode)"""
    print(f"  📏 Generating Caption Length Comparison ({mode})...")

    all_model_lengths = []
    gt_lens = None

    for i, model_info in enumerate(models_data):
        path = model_info["path"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"    ⚠️  Failed to read {path}: {e}")
            continue

        pred_lens, cur_gt_lens = _extract_caption_lengths_from_report(d)
        if pred_lens:
            all_model_lengths.append({
                "name": model_info["model"],
                "data": pred_lens
            })

        if gt_lens is None and cur_gt_lens:
            gt_lens = cur_gt_lens

    if not all_model_lengths:
        print(f"    ⚠️  No caption lengths found for mode '{mode}'.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_colors = ['#2E86AB', '#E63946', '#06A77D', '#F18F01', '#A23B72']
    gt_color = '#34495e'

    stats_text = "Summary Statistics:\n"
    max_len_global = 0

    if gt_lens and len(gt_lens) > 3:
        gt_mean = np.mean(gt_lens)
        max_len_global = max(max_len_global, max(gt_lens))
        
        if HAS_SEABORN:
            sns.kdeplot(gt_lens, color=gt_color, 
                       label=f'Ground Truth (μ={gt_mean:.1f})', 
                       fill=True, alpha=0.25, linewidth=3, 
                       linestyle='--', ax=ax, warn_singular=False)
        else:
            ax.axvline(gt_mean, color=gt_color, linestyle='--', 
                      linewidth=2.5, label=f'GT Mean ({gt_mean:.1f})')
        
        stats_text += f"Ground Truth: {gt_mean:.1f} words\n"

    for i, model in enumerate(all_model_lengths):
        color = model_colors[i % len(model_colors)]
        data = model["data"]
        name = model["name"]
        
        if len(data) < 3: continue
        
        mean = np.mean(data)
        max_len_global = max(max_len_global, max(data))

        if HAS_SEABORN:
            sns.kdeplot(data, color=color, 
                       label=f'{name} (μ={mean:.1f})', 
                       fill=True, alpha=0.3, linewidth=2.5, 
                       ax=ax, warn_singular=False)
        else:
            ax.hist(data, bins=20, alpha=0.4, color=color, 
                   label=f'{name} (μ={mean:.1f})',
                   density=True, edgecolor='black')
        
        stats_text += f"{name}: {mean:.1f} words\n"

    ax.set_title(f"Caption Length Comparison: Model Verbosity Analysis ({mode.upper()})", 
                fontsize=18, fontweight="bold", pad=25)
    ax.set_xlabel("Caption Length (words)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Density", fontsize=14, fontweight="bold")
    
    limit_len = min(max_len_global + 5, 50)
    ax.set_xlim(0, limit_len)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"caption_length_comparison_{mode}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Saved comparison: {out_path}")

def compare_models(results_paths: List[str], output_dir: str):
    """Generate comparison plots, separated by evaluation mode."""
    print(f"\n⚖️  Generating Model Comparison Plots...")

    by_mode = {}
    for path in results_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            mode = d.get('mode', 'unknown')
            by_mode.setdefault(mode, []).append({
                'model': str(d.get('model', 'unknown')).upper(),
                'summary': d.get('summary', d.get('metrics', {})),
                'path': path
            })
        except Exception as e:
            print(f"  ⚠️  Error loading {path}: {e}")

    if not by_mode:
        print("  ❌ No valid data found")
        return

    for mode, items in by_mode.items():
        if len(items) < 2:
            print(f"  ⚠️  Skipping mode '{mode}': need at least 2 result files to compare.")
            continue
        _compare_models_single_mode(items, mode, output_dir)
        _compare_caption_lengths_single_mode(items, mode, output_dir)
