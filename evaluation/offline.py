import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
from colorama import Fore

from src.config import Config
from evaluation.metrics import MetricsCalculator, hungarian_matching

CURVE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def _scene_fields(scene: dict):
    start = scene.get("start", scene.get("Start"))
    end = scene.get("end", scene.get("End"))
    caption = scene.get("caption", scene.get("Description", scene.get("text", "")))
    return start, end, caption

def evaluate_single_file(pred_data, gt_entry, metrics_calc, iou_thresh):
    gt_timestamps = gt_entry['timestamps']
    gt_sentences = gt_entry['sentences']
    predicted_scenes = pred_data.get('scenes', [])

    pred_intervals = []
    pred_texts = []
    for s in predicted_scenes:
        start, end, caption = _scene_fields(s)
        if start is None or end is None:
            continue
        pred_intervals.append((float(start), float(end)))
        pred_texts.append(str(caption))

    # Hungarian Matching
    matches = hungarian_matching(pred_intervals, gt_timestamps, iou_thresh)

    # Curve stats
    curve_stats = []
    for t in CURVE_THRESHOLDS:
        m_t = hungarian_matching(pred_intervals, gt_timestamps, t)
        gt_matched_t = len({gt_idx for gt_idx, _ in m_t.values()})
        curve_stats.append({
            "iou": t,
            "matches": int(len(m_t)),
            "gt_matched": int(gt_matched_t),
            "pred": int(len(pred_intervals)),
            "gt": int(len(gt_timestamps))
        })

    logs = []
    matched_scores = []
    
    # Store tokens for corpus calculation
    matched_refs_tokens = []
    matched_pred_tokens = []

    gt_matched_indices = set()

    for pred_idx, pred_int in enumerate(pred_intervals):
        status = "FALSE POSITIVE"
        scores = {"BLEU_3": 0.0, "BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0}
        gt_text_disp = "N/A"
        iou_value = 0.0

        if pred_idx in matches:
            gt_idx, matched_iou = matches[pred_idx]
            status = "HIT"
            gt_text_raw = gt_sentences[gt_idx]
            
            if isinstance(gt_text_raw, list) and gt_text_raw:
                gt_text_disp = gt_text_raw[0]
            else:
                gt_text_disp = gt_text_raw
            
            # --- Sentence Level Calculation (For Logs) ---
            scores = metrics_calc.compute(gt_text_raw, pred_texts[pred_idx])
            matched_scores.append(scores)
            
            # --- Token Collection (For Corpus) ---
            # Prepare Reference: List[List[str]]
            if isinstance(gt_text_raw, list):
                ref_list = [r.lower().split() for r in gt_text_raw]
            else:
                ref_list = [str(gt_text_raw).lower().split()]
            
            # Prepare Candidate: List[str]
            cand_list = pred_texts[pred_idx].lower().split()
            
            matched_refs_tokens.append(ref_list)
            matched_pred_tokens.append(cand_list)

            gt_matched_indices.add(gt_idx)
            iou_value = matched_iou

        logs.append({
            "pred_time": pred_int,
            "iou": round(float(iou_value), 2),
            "status": status,
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "pred": pred_texts[pred_idx],
            "gt": gt_text_disp
        })

    for gt_idx, gt_timestamp in enumerate(gt_timestamps):
        if gt_idx not in gt_matched_indices:
            gt_text = gt_sentences[gt_idx]
            if isinstance(gt_text, list) and gt_text:
                gt_text = gt_text[0]
            else:
                gt_text = str(gt_text)
            
            logs.append({
                "gt_time": gt_timestamp,
                "iou": 0.0,
                "status": "FALSE NEGATIVE",
                "scores": {"BLEU_3": 0.0, "BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0},
                "pred": "N/A",
                "gt": gt_text
            })

    precision = len(matches) / len(pred_intervals) if pred_intervals else 0
    recall = len(gt_matched_indices) / len(gt_timestamps) if gt_timestamps else 0

    return {
        "precision": precision,
        "recall": recall,
        "matched_scores": matched_scores,
        "matched_count": len(matches),
        "total_predictions": len(pred_intervals),
        "total_gt": len(gt_timestamps),
        "curve_stats": curve_stats,
        "logs": logs,
        "matched_refs_tokens": matched_refs_tokens, 
        "matched_pred_tokens": matched_pred_tokens  
    }

def run(args):
    metrics_calc = MetricsCalculator()
    results_dir = os.path.join(Config.RESULTS_DIR, args.model)
    
    print(Fore.CYAN + f"📂 Ground Truth: {args.json}")
    print(Fore.CYAN + f"📂 Results Dir:  {results_dir}")

    if not os.path.exists(results_dir):
        print(Fore.RED + "❌ Results folder not found.")
        return

    try:
        with open(args.json, 'r', encoding='utf-8') as f: 
            gt_data = json.load(f)
    except Exception as e:
        print(Fore.RED + f"❌ Error reading GT: {e}")
        return

    search_pattern = os.path.join(results_dir, "**", "*.json")
    json_files = glob(search_pattern, recursive=True)
    json_files = [f for f in json_files 
                  if "summary" not in f and "pipeline_results" not in f]

    if not json_files:
        print(Fore.RED + "❌ No JSON files found.")
        return
    
    if args.limit:
        json_files = json_files[:args.limit]

    print(Fore.YELLOW + f"🔍 Found {len(json_files)} files for evaluation...")
    
    global_stats = {
        "precision_per_video": [],
        "recall_per_video": [],
        "all_matched_scores": []
    }
    
    # Corpus Accumulators
    corpus_references = []
    corpus_candidates = []
    
    all_logs = {}
    processed_count = 0
    total_matched = 0
    total_predictions = 0
    total_gt = 0
    curve_acc = {t: {"matches": 0, "gt_matched": 0, "pred": 0, "gt": 0} for t in CURVE_THRESHOLDS}

    for pred_file in tqdm(json_files, unit="vid", desc="Offline Eval"):
        filename = os.path.basename(pred_file)
        vid_id = os.path.splitext(filename)[0]
        
        gt_entry = None
        if vid_id in gt_data: 
            gt_entry = gt_data[vid_id]
        elif f"v_{vid_id}" in gt_data: 
            gt_entry = gt_data[f"v_{vid_id}"]
        elif vid_id.startswith("v_") and vid_id[2:] in gt_data: 
            gt_entry = gt_data[vid_id[2:]]
            
        if not gt_entry: 
            continue
            
        try:
            with open(pred_file, 'r', encoding='utf-8') as f: 
                pred_data = json.load(f)
            
            res = evaluate_single_file(pred_data, gt_entry, metrics_calc, args.threshold)
            
            global_stats["precision_per_video"].append(res["precision"])
            global_stats["recall_per_video"].append(res["recall"])
            global_stats["all_matched_scores"].extend(res["matched_scores"])
            
            # Aggregate Corpus Data
            corpus_references.extend(res["matched_refs_tokens"])
            corpus_candidates.extend(res["matched_pred_tokens"])
            
            total_matched += res["matched_count"]
            total_predictions += res["total_predictions"]
            total_gt += res["total_gt"]

            all_logs[vid_id] = res

            for cs in res.get("curve_stats", []):
                t = float(cs["iou"])
                curve_acc[t]["matches"] += int(cs["matches"])
                curve_acc[t]["gt_matched"] += int(cs["gt_matched"])
                curve_acc[t]["pred"] += int(cs["pred"])
                curve_acc[t]["gt"] += int(cs["gt"])

            processed_count += 1
            
        except Exception as e:
            print(Fore.RED + f"⚠️  Error processing {filename}: {e}")
            continue

    print(Fore.MAGENTA + "="*60)
    print(Fore.GREEN + f"📊 OFFLINE REPORT: {args.model.upper()} | Hungarian")
    print(Fore.CYAN + f"   Videos Processed:  {processed_count}")
    print(Fore.CYAN + f"   Total Matches:     {total_matched}")
    
    final_results = {}
    
    if processed_count > 0:
        micro_precision = (total_matched / total_predictions) * 100 if total_predictions else 0.0
        micro_recall = (total_matched / total_gt) * 100 if total_gt else 0.0

        final_results["averaging"] = "micro"
        final_results["precision"] = round(micro_precision, 2)
        final_results["recall"] = round(micro_recall, 2)

        temporal_curve = []
        for t in CURVE_THRESHOLDS:
            denom_pred = curve_acc[t]["pred"]
            denom_gt = curve_acc[t]["gt"]
            p = (curve_acc[t]["matches"] / denom_pred) * 100 if denom_pred else 0.0
            r = (curve_acc[t]["gt_matched"] / denom_gt) * 100 if denom_gt else 0.0
            temporal_curve.append({"iou": t, "precision": round(p, 2), "recall": round(r, 2)})
        final_results["temporal_curve"] = temporal_curve
        
        print(Fore.YELLOW + f"   - Precision:  {micro_precision:.2f}%")
        print(Fore.YELLOW + f"   - Recall:     {micro_recall:.2f}%")
        
        # --- Metrics Calculation ---
        if corpus_candidates:
            # 1. Corpus BLEU
            c_bleu3, c_bleu4 = metrics_calc.compute_corpus_bleu(corpus_references, corpus_candidates)
            
            # 2. METEOR/ROUGE
            all_meteor = [s["METEOR"] for s in global_stats["all_matched_scores"]]
            all_rouge = [s["ROUGE_L"] for s in global_stats["all_matched_scores"]]
            
            final_results["BLEU_3"] = round(c_bleu3, 2)
            final_results["BLEU_4"] = round(c_bleu4, 2)
            final_results["METEOR"] = round(np.mean(all_meteor) * 100, 2)
            final_results["ROUGE_L"] = round(np.mean(all_rouge) * 100, 2)
            
            print(Fore.YELLOW + f"   - BLEU-3 (Corpus): {final_results['BLEU_3']:.2f}%")
            print(Fore.YELLOW + f"   - BLEU-4 (Corpus): {final_results['BLEU_4']:.2f}%")
            print(Fore.YELLOW + f"   - METEOR:          {final_results['METEOR']:.2f}%")
            print(Fore.YELLOW + f"   - ROUGE_L:         {final_results['ROUGE_L']:.2f}%")
        else:
            final_results.update({"BLEU_3": 0.0, "BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"offline_report_{args.model}_{timestamp}.json"
    report_path = os.path.join(Config.EVAL_OFFLINE_DIR, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "mode": "offline",
            "threshold": args.threshold,
            "matching": "hungarian",
            "timestamp": timestamp,
            "total_videos": processed_count,
            "summary": final_results,
            "details": all_logs
        }, f, indent=4)
        
    print(Fore.CYAN + f"💾 Saved: {report_path}")