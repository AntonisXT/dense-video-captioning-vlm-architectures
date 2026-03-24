import os
import json
import numpy as np
from tqdm import tqdm
from colorama import Fore

from src.config import Config
from src.model_factory import get_caption_engine
from evaluation.metrics import MetricsCalculator

def run(args):
    output_filename = f"oracle_results_{args.model}.json"
    output_path = os.path.join(Config.EVAL_ORACLE_DIR, output_filename)
    
    print(Fore.CYAN + f"📂 Output File: {output_path}")

    if not os.path.exists(args.json):
        print(Fore.RED + "❌ Ground Truth not found.")
        return
    with open(args.json, 'r', encoding='utf-8') as f: gt_data = json.load(f)
    if not os.path.exists(args.videos):
        print(Fore.RED + "❌ Videos directory not found.")
        return
    
    avail = {os.path.splitext(f)[0]: f for f in os.listdir(args.videos)}
    eval_pairs = []
    for key in gt_data.keys():
        yid = key[2:] if key.startswith("v_") else key
        if yid in avail: eval_pairs.append((key, avail[yid]))
    if args.limit: eval_pairs = eval_pairs[:args.limit]
    
    print(Fore.YELLOW + f"🔍 Oracle Eval: {len(eval_pairs)} videos")

    try:
        caption_engine = get_caption_engine(args.model)
        metrics_calc = MetricsCalculator()
    except Exception as e:
        print(Fore.RED + f"Error loading model: {e}")
        return

    all_matched_scores = []
    detailed_results = {}
    
    # Corpus Accumulators
    corpus_references = []
    corpus_candidates = []
    
    total_timestamps = 0
    successful_captions = 0
    failed_extractions = 0
    
    for vid_id, vid_filename in tqdm(eval_pairs, desc="Oracle Eval", unit="vid"):
        video_path = os.path.join(args.videos, vid_filename)
        entry = gt_data[vid_id]
        timestamps = entry.get('timestamps', [])
        gt_sentences = entry.get('sentences', [])
        
        if not timestamps: continue
        total_timestamps += len(timestamps)
        
        try:
            visual_inputs = caption_engine.extract_frames_batch(video_path, timestamps)
        except Exception:
            failed_extractions += len(timestamps)
            continue
            
        vid_results = []
        for i, (vis_input, gt_sent) in enumerate(zip(visual_inputs, gt_sentences)):
            if vis_input is None:
                failed_extractions += 1
                continue
            try:
                gen_cap = caption_engine.generate_caption(vis_input)
                
                # 1. Sentence Scores (For Logs)
                scores = metrics_calc.compute(gt_sent, gen_cap)
                all_matched_scores.append(scores)
                successful_captions += 1
                
                # 2. Token Collection (For Corpus)
                if isinstance(gt_sent, list):
                    ref_list = [r.lower().split() for r in gt_sent]
                else:
                    ref_list = [str(gt_sent).lower().split()]
                
                cand_list = gen_cap.lower().split()
                
                corpus_references.append(ref_list)
                corpus_candidates.append(cand_list)
                
                vid_results.append({
                    "timestamp": timestamps[i],
                    "generated": gen_cap,
                    "ground_truth": gt_sent,
                    "scores": {k: round(v, 4) for k, v in scores.items()}
                })
            except Exception as e:
                failed_extractions += 1
                continue
        
        if vid_results:
            detailed_results[vid_id] = vid_results

    # --- REPORTING ---
    print(Fore.MAGENTA + "\n" + "="*60)
    print(Fore.GREEN + f"📊 ORACLE RESULTS ({args.model.upper()})")
    print(Fore.CYAN + f"   Total GT Timestamps: {total_timestamps}")
    
    final_avgs = {}
    if corpus_candidates:
        # Corpus BLEU
        c_bleu3, c_bleu4 = metrics_calc.compute_corpus_bleu(corpus_references, corpus_candidates)
        
        # Sentence Average for others
        all_meteor = [s["METEOR"] for s in all_matched_scores]
        all_rouge = [s["ROUGE_L"] for s in all_matched_scores]
        
        final_avgs["BLEU_3"] = c_bleu3
        final_avgs["BLEU_4"] = c_bleu4
        final_avgs["METEOR"] = np.mean(all_meteor) * 100
        final_avgs["ROUGE_L"] = np.mean(all_rouge) * 100
        
        print(Fore.YELLOW + f"   - BLEU-3 (Corpus): {final_avgs['BLEU_3']:.2f}%")
        print(Fore.YELLOW + f"   - BLEU-4 (Corpus): {final_avgs['BLEU_4']:.2f}%")
        print(Fore.YELLOW + f"   - METEOR:          {final_avgs['METEOR']:.2f}%")
        print(Fore.YELLOW + f"   - ROUGE_L:         {final_avgs['ROUGE_L']:.2f}%")
    else:
        print(Fore.RED + "   ❌ No successful captions to evaluate!")
        
    print(Fore.MAGENTA + "="*60)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "mode": "oracle",
            "summary": {k: round(v, 2) for k, v in final_avgs.items()},
            "details": detailed_results
        }, f, indent=4, ensure_ascii=False)
    
    print(Fore.CYAN + f"💾 Saved: {output_path}")