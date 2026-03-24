import torch
from sentence_transformers import SentenceTransformer, util
from colorama import Fore, Style
from src.config import Config
from src.logger import log

class SceneMerger:
    """
    Handles the semantic merging of consecutive video scenes based on caption similarity.
    It leverages a Sentence Transformer model to compute semantic embeddings and determines
    whether adjacent scenes describe the same event.
    """

    def __init__(self, threshold: float = Config.MERGE_THRESHOLD):
        """
        Initialize the merger with a specific similarity threshold.
        
        Args:
            threshold (float): The cosine similarity threshold required to merge two scenes.
                               Values range from -1.0 to 1.0.
        """
        self.threshold = threshold
        log.info(f"🧠 Loading semantic model ({Config.SIMILARITY_MODEL})...")
        self.model = SentenceTransformer(Config.SIMILARITY_MODEL, device=Config.DEVICE)

    def _similarity(self, text_a: str, text_b: str) -> float:
        """Computes the cosine similarity between two text descriptions."""
        embeddings = self.model.encode([text_a, text_b], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    def _pick_best_caption(self, scenes_buffer: list, previous_caption: str = None, candidate_embeddings=None) -> str:
        """
        Selects the most representative caption for a merged group of scenes.
        
        Selection Strategy:
            1. Centroid Score: Measures how well a caption represents the current group (average similarity to others).
            2. Context Score: Measures logical continuity with the previous scene's caption.
            
            - If a previous caption exists, a weighted combination (70% Centroid, 30% Context) is used.
            - Otherwise, pure Centroid selection is applied.
        
        Args:
            scenes_buffer (list): List of scene dictionaries to be merged.
            previous_caption (str, optional): The final caption of the immediately preceding scene.
            candidate_embeddings (Tensor, optional): Precomputed embeddings for efficiency.
            
        Returns:
            str: The selected best caption.
        """
        if not scenes_buffer: return ""
        if len(scenes_buffer) == 1: return scenes_buffer[0]['Description']

        candidates = [s['Description'] for s in scenes_buffer]
        embeddings = candidate_embeddings
        if embeddings is None:
            embeddings = self.model.encode(candidates, convert_to_tensor=True)

        # 1. Compute Centroid Scores
        # We calculate the average cosine similarity of each candidate against all others in the group.
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        centroid_scores = torch.sum(cosine_scores, dim=1)
        centroid_scores = centroid_scores / len(scenes_buffer)

        if previous_caption:
            # 2. Compute Context Scores
            # We calculate the cosine similarity with the previous scene's caption.
            prev_emb = self.model.encode(previous_caption, convert_to_tensor=True)
            context_scores = util.pytorch_cos_sim(prev_emb, embeddings)[0]
            
            # Weighted Selection: 70% Importance to Group Representativeness, 30% to Contextual Flow.
            combined_scores = 0.7 * centroid_scores + 0.3 * context_scores
            
            best_idx = torch.argmax(combined_scores).item()
            best_centroid = centroid_scores[best_idx].item()
            best_context = context_scores[best_idx].item()
            
            log.info(Fore.YELLOW + f"         🧠 Weighted Selection (Centroid: {best_centroid:.2f}, Context: {best_context:.2f})")
            return candidates[best_idx]
        
        # Fallback: Pure Centroid Selection (First scene or no context available)
        best_idx = torch.argmax(centroid_scores).item()
        log.info(Fore.YELLOW + f"         🎯 Centroid Selection (Score: {centroid_scores[best_idx]:.2f})")
        return candidates[best_idx]

    def merge_scenes(self, results: list) -> list:
        """
        Iteratively merges consecutive scenes if their descriptions are semantically similar.

        Optimization:
            - Embeddings are precomputed for all scenes in a batch to minimize model inference time.
            - Pairwise comparisons are performed using these precomputed embeddings.
            
        Args:
            results (list): List of dictionaries containing raw scene data (Start, End, Description).
            
        Returns:
            list: A new list of merged scene dictionaries.
        """
        if not results:
            return []

        log.info(f"   🔄 Starting scene merging (Threshold: {self.threshold})...")

        # Precompute embeddings for all descriptions in one batch
        descriptions = [r.get('Description', '') for r in results]
        all_embeddings = self.model.encode(descriptions, convert_to_tensor=True)

        merged_results = []
        buffer = [results[0]]
        buffer_start_idx = 0
        last_finalized_desc = None

        # Iterate through scenes and compare adjacent pairs
        for i in range(1, len(results)):
            current_scene = results[i]
            sim = util.pytorch_cos_sim(all_embeddings[i - 1], all_embeddings[i]).item()

            # Logging: Visual indicator of similarity
            # Cyan '🔗' indicates a merge will happen; Dimmed output otherwise.
            symbol = "🔗" if sim >= self.threshold else "  "
            color = Fore.CYAN if sim >= self.threshold else Style.DIM
            
            log.info(
                color + f"      {symbol} S{results[i - 1]['Scene ID']} vs S{current_scene['Scene ID']} | Sim: {sim:.2f}"
            )

            if sim >= self.threshold:
                # Similarity is high enough; add to buffer and continue
                buffer.append(current_scene)
                continue

            # Similarity threshold not met; finalize the current buffer
            cand_emb = all_embeddings[buffer_start_idx:i]
            best_desc = self._pick_best_caption(
                buffer,
                previous_caption=last_finalized_desc,
                candidate_embeddings=cand_emb,
            )

            merged_scene = {
                "Scene ID": buffer[0]['Scene ID'],
                "Start": buffer[0]['Start'],
                "End": buffer[-1]['End'],
                "Duration": round(buffer[-1]['End'] - buffer[0]['Start'], 2),
                "Description": best_desc,
            }
            merged_results.append(merged_scene)

            # Reset buffer for the next group
            last_finalized_desc = best_desc
            buffer = [current_scene]
            buffer_start_idx = i

        # Finalize any remaining scenes in the buffer
        if buffer:
            cand_emb = all_embeddings[buffer_start_idx:len(results)]
            best_desc = self._pick_best_caption(
                buffer,
                previous_caption=last_finalized_desc,
                candidate_embeddings=cand_emb,
            )
            merged_scene = {
                "Scene ID": buffer[0]['Scene ID'],
                "Start": buffer[0]['Start'],
                "End": buffer[-1]['End'],
                "Duration": round(buffer[-1]['End'] - buffer[0]['Start'], 2),
                "Description": best_desc,
            }
            merged_results.append(merged_scene)

        log.info(f"   ✅ Merging complete: {len(results)} raw -> {len(merged_results)} merged scenes.")
        return merged_results