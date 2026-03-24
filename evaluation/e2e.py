import os
import json
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

from src.config import Config
from src.logger import log
from src.model_factory import get_caption_engine
from src.scene_engine import SceneEngine
from src.merger import SceneMerger
from evaluation.metrics import MetricsCalculator, hungarian_matching

CURVE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def _scene_fields(scene: dict):
    start = scene.get("start", scene.get("Start"))
    end = scene.get("end", scene.get("End"))
    caption = scene.get("caption", scene.get("Description", scene.get("text", "")))
    return start, end, caption

def evaluate_video(video_path, gt_timestamps, gt_sentences, caption_engine,
                   scene_merger, metrics_calc, iou_thresh):
    
    # 1. Scene Detection
    scene_engine = SceneEngine(video_path)
    detected_scenes = [s for s in scene_engine.detect_scenes() if (s[1] - s[0]) >= 0.2]

    # 2. Frame Extraction & Captioning
    visual_inputs = caption_engine.extract_frames_batch(video_path, detected_scenes)
    raw_results = []

    log.info(Fore.MAGENTA + f"\n🎥 Analyzing: {os.path.basename(video_path)}")
    
    for i, (scene, vis_input) in enumerate(zip(detected_scenes, visual_inputs)):
        if vis_input is not None:
            desc = caption_engine.generate_caption(vis_input)
            log.info(Fore.CYAN + f"   ⏱️  {scene[0]:.1f}-{scene[1]:.1f}s: " + Fore.WHITE + f"{desc}")
            raw_results.append({
                "Scene ID": i + 1,
                "Start": scene[0],
                "End": scene[1],
                "Description": desc
            })

    # 3. Scene Merging (Ablation Check)
    if Config.ENABLE_SCENE_MERGING and raw_results:
        final_scenes = scene_merger.merge_scenes(raw_results)
    else:
        final_scenes = raw_results

    # 4. Evaluation
    pred_intervals = []
    pred_texts = []
    for s in final_scenes:
        start, end, caption = _scene_fields(s)
        if start is None or end is None: continue
        pred_intervals.append((float(start), float(end)))
        pred_texts.append(str(caption))

    matches = hungarian_matching(pred_intervals, gt_timestamps, iou_thresh)

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

    matched_scores = []
    
    # Token collection for Corpus
    matched_refs_tokens = []
    matched_pred_tokens = []
    
    gt_matched_indices = set()
    logs = []

    for pred_idx, pred_int in enumerate(pred_intervals):
        status = "FALSE POSITIVE"
        scores = {"BLEU_3": 0.0, "BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0}
        gt_text = "N/A"
        iou_value = 0.0

        if pred_idx in matches:
            gt_idx, matched_iou = matches[pred_idx]
            status = "HIT"
            gt_text = gt_sentences[gt_idx]
            
            if isinstance(gt_text, list) and gt_text:
                gt_text = gt_text[0]
            
            # Sentence Scores
            scores = metrics_calc.compute(gt_text, pred_texts[pred_idx])
            matched_scores.append(scores)
            
            # Token Collection
            if isinstance(gt_text, list):
                ref_list = [r.lower().split() for r in gt_text]
            else:
                ref_list = [str(gt_text).lower().split()]
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
            "gt": gt_text
        })

    for gt_idx, gt_timestamp in enumerate(gt_timestamps):
        if gt_idx not in gt_matched_indices:
            gt_text = gt_sentences[gt_idx]
            if isinstance(gt_text, list) and gt_text:
                gt_text = gt_text[0]
            
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
        "logs": logs,
        "curve_stats": curve_stats,
        "matched_refs_tokens": matched_refs_tokens,
        "matched_pred_tokens": matched_pred_tokens
    }

def run(args):
    output_filename = f"e2e_results_{args.model}.json"
    output_file = os.path.join(Config.EVAL_E2E_DIR, output_filename)
    log.info(Fore.CYAN + f"📂 Output will be saved to: {output_file}")

    if not os.path.exists(args.json): 
        log.error("❌ Ground Truth file not found")
        return
    with open(args.json, 'r', encoding='utf-8') as f: gt_data = json.load(f)
    
    avail = {os.path.splitext(f)[0]: f for f in os.listdir(args.videos)}
    pairs = []
    for k in gt_data:
        yid = k[2:] if k.startswith("v_") else k
        if yid in avail: pairs.append((k, avail[yid]))
    if args.limit: pairs = pairs[:args.limit]

    caption_engine = get_caption_engine(args.model)
    merger = SceneMerger() 
    metrics = MetricsCalculator()

    global_stats = {
        "precision_per_video": [],
        "recall_per_video": [],
        "all_matched_scores": []
    }
    
    corpus_references = []
    corpus_candidates = []
    
    all_logs = {}
    total_matched = 0
    total_predictions = 0
    total_gt = 0
    curve_acc = {t: {"matches": 0, "gt_matched": 0, "pred": 0, "gt": 0} for t in CURVE_THRESHOLDS}

    log.info(Fore.YELLOW + "🔍 Starting Live Evaluation (Hungarian Matching)...")
    
    for i, (vid_id, fname) in enumerate(pairs):
        log.info(Fore.MAGENTA + f"\n[{i+1}/{len(pairs)}] Processing {fname}...")
        path = os.path.join(args.videos, fname)
        entry = gt_data[vid_id]
        
        res = evaluate_video(
            path, entry['timestamps'], entry['sentences'], 
            caption_engine, merger, metrics, args.threshold
        )
        
        global_stats["precision_per_video"].append(res["precision"])
        global_stats["recall_per_video"].append(res["recall"])
        global_stats["all_matched_scores"].extend(res["matched_scores"])
        
        corpus_references.extend(res["matched_refs_tokens"])
        corpus_candidates.extend(res["matched_pred_tokens"])
        
        total_matched += res["matched_count"]
        total_predictions += res["total_predictions"]
        total_gt += res["total_gt"]

        for cs in res.get("curve_stats", []):
            t = float(cs["iou"])
            curve_acc[t]["matches"] += int(cs["matches"])
            curve_acc[t]["gt_matched"] += int(cs["gt_matched"])
            curve_acc[t]["pred"] += int(cs["pred"])
            curve_acc[t]["gt"] += int(cs["gt"])
        
        all_logs[vid_id] = res

    log.info(Fore.MAGENTA + "="*60)
    log.info(Fore.GREEN + f"📊 E2E RESULTS ({args.model}) | IoU > {args.threshold} | Hungarian")
    
    final_summary = {}
    micro_precision = (total_matched / total_predictions) * 100 if total_predictions else 0.0
    micro_recall = (total_matched / total_gt) * 100 if total_gt else 0.0

    final_summary["averaging"] = "micro"
    final_summary["precision"] = round(micro_precision, 2)
    final_summary["recall"] = round(micro_recall, 2)

    temporal_curve = []
    for t in CURVE_THRESHOLDS:
        denom_pred = curve_acc[t]["pred"]
        denom_gt = curve_acc[t]["gt"]
        p = (curve_acc[t]["matches"] / denom_pred) * 100 if denom_pred else 0.0
        r = (curve_acc[t]["gt_matched"] / denom_gt) * 100 if denom_gt else 0.0
        temporal_curve.append({"iou": t, "precision": round(p, 2), "recall": round(r, 2)})
    final_summary["temporal_curve"] = temporal_curve
    
    log.info(Fore.YELLOW + f"   - Precision:  {micro_precision:.2f}%")
    log.info(Fore.YELLOW + f"   - Recall:     {micro_recall:.2f}%")
    
    if corpus_candidates:
        c_bleu3, c_bleu4 = metrics.compute_corpus_bleu(corpus_references, corpus_candidates)
        all_meteor = [s["METEOR"] for s in global_stats["all_matched_scores"]]
        all_rouge = [s["ROUGE_L"] for s in global_stats["all_matched_scores"]]
        
        final_summary["BLEU_3"] = round(c_bleu3, 2)
        final_summary["BLEU_4"] = round(c_bleu4, 2)
        final_summary["METEOR"] = round(np.mean(all_meteor) * 100, 2)
        final_summary["ROUGE_L"] = round(np.mean(all_rouge) * 100, 2)
        
        log.info(Fore.YELLOW + f"   - BLEU-3 (Corpus): {final_summary['BLEU_3']:.2f}%")
        log.info(Fore.YELLOW + f"   - BLEU-4 (Corpus): {final_summary['BLEU_4']:.2f}%")
        log.info(Fore.YELLOW + f"   - METEOR:          {final_summary['METEOR']:.2f}%")
        log.info(Fore.YELLOW + f"   - ROUGE_L:         {final_summary['ROUGE_L']:.2f}%")
    else:
        final_summary.update({"BLEU_3": 0.0, "BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0})
        log.error("   ⚠️  No matches found - all metrics are 0.0")
    
    log.info(Fore.MAGENTA + "="*60)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "mode": "e2e",
            "threshold": args.threshold,
            "matching": "hungarian",
            "total_videos": len(pairs),
            "summary": final_summary, 
            "details": all_logs
        }, f, indent=4, ensure_ascii=False)
    
    log.info(Fore.CYAN + f"💾 Results saved: {output_file}")