import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment 
from typing import Dict, Tuple, List

def compute_meteor_safe(references, candidate: str) -> float:
    """
    Robust METEOR across NLTK versions with support for multi-reference.
    """
    if references is None:
        references_list = []
    elif isinstance(references, (list, tuple)):
        references_list = [str(r) for r in references if str(r).strip()]
    else:
        references_list = [str(references)] if str(references).strip() else []

    cand = (candidate or "").strip().lower()
    if not references_list or not cand:
        return 0.0

    refs_norm = [r.strip().lower() for r in references_list if r.strip()]
    if not refs_norm:
        return 0.0

    try:
        return float(meteor_score(refs_norm, cand))
    except TypeError:
        pass
    except Exception as e:
        if "pre-tokenized" not in str(e):
            raise

    refs_tokens = [r.split() for r in refs_norm if r.split()]
    cand_tokens = cand.split()
    if not refs_tokens or not cand_tokens:
        return 0.0

    try:
        return float(meteor_score(refs_tokens, cand_tokens))
    except Exception:
        return 0.0

def calculate_iou(interval_1: Tuple[float, float], interval_2: Tuple[float, float]) -> float:
    start1, end1 = interval_1
    start2, end2 = interval_2
    intersection = max(0.0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    if union <= 0: return 0.0
    return intersection / union

def hungarian_matching(
    pred_intervals: List[Tuple[float, float]],
    gt_intervals: List[Tuple[float, float]],
    iou_threshold: float,
) -> Dict[int, Tuple[int, float]]:
    """
    Optimal bipartite matching using the Hungarian Algorithm.
    Corrected to penalized invalid matches during assignment.
    """
    num_pred = len(pred_intervals)
    num_gt = len(gt_intervals)
    
    if num_pred == 0 or num_gt == 0:
        return {}

    # Create IoU matrix (Rows: Preds, Cols: GT)
    iou_matrix = np.zeros((num_pred, num_gt))
    for i, p in enumerate(pred_intervals):
        for j, g in enumerate(gt_intervals):
            iou_matrix[i, j] = calculate_iou(p, g)

    # Scipy minimizes cost. We usually use negative IoU (-0.9 is better than -0.1).
    cost_matrix = -iou_matrix
    
    # Penalize matches below threshold so the algorithm avoids them 
    # unless absolutely necessary (effectively prioritizing valid matches).
    # Since valid costs are between -1.0 and 0.0, a large positive cost (+100)
    cost_matrix[iou_matrix < iou_threshold] = 100.0
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: Dict[int, Tuple[int, float]] = {}
    for r, c in zip(row_ind, col_ind):
        # We still double-check, but now the algorithm prioritized valid pairs
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            matches[r] = (c, float(iou))
            
    return matches

class MetricsCalculator:
    """
    Centralized NLP metrics calculator.
    """

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4

    def compute(self, reference, candidate: str) -> Dict[str, float]:
        """
        Computes Sentence-Level BLEU (for logs), METEOR, and ROUGE-L.
        """
        if reference is None:
            references = []
        elif isinstance(reference, (list, tuple)):
            references = [str(r) for r in reference if str(r).strip()]
        else:
            references = [str(reference)] if str(reference).strip() else []

        candidate = candidate or ""
        ref_token_lists = [r.lower().split() for r in references if r.strip()]
        cand_tokens = candidate.lower().split()

        has_tokens = (ref_token_lists and cand_tokens)

        # BLEU-3 (Weights: 1/3 each for 1,2,3-grams)
        if has_tokens:
            bleu3 = sentence_bleu(
                ref_token_lists, cand_tokens,
                weights=(1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0),
                smoothing_function=self.smoothie
            )
        else:
            bleu3 = 0.0

        # BLEU-4 (Weights: 1/4 each for 1,2,3,4-grams - Standard)
        if has_tokens:
            bleu4 = sentence_bleu(
                ref_token_lists, cand_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothie
            )
        else:
            bleu4 = 0.0

        meteor = compute_meteor_safe(references, candidate)

        if references and candidate.strip():
            rouge_l = max(self.rouge.score(ref, candidate)["rougeL"].fmeasure for ref in references)
        else:
            rouge_l = 0.0

        return {
            "BLEU_3": float(bleu3),
            "BLEU_4": float(bleu4),
            "METEOR": float(meteor),
            "ROUGE_L": float(rouge_l),
        }

    def compute_corpus_bleu(self, all_references, all_candidates) -> Tuple[float, float]:
        """
        Computes Corpus-Level BLEU (Standard for reporting).
        
        Args:
            all_references: List[List[List[str]]] (List of test cases, each has list of refs, each ref is list of tokens)
            all_candidates: List[List[str]] (List of test cases, each has list of tokens)
        """
        if not all_references or not all_candidates:
            return 0.0, 0.0

        # BLEU-3
        b3 = corpus_bleu(
            all_references, all_candidates,
            weights=(1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0),
            smoothing_function=self.smoothie
        )

        # BLEU-4
        b4 = corpus_bleu(
            all_references, all_candidates,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothie
        )

        return b3 * 100.0, b4 * 100.0