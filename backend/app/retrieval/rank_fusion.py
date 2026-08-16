from typing import List, Dict, Any

class ReciprocalRankFusion:
    """Combines ranked candidate lists from multiple modalities using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates fused score for each unique entity (e.g. keyframe or second timestamp).
        
        Formula:
        RRF(d) = sum_{m in M} ( w_m / (k + rank_m(d)) )
        
        Returns: dict of {item_id: {"item_data": ..., "fused_score": float, "modality_scores": ...}}
        """
        fused_results: Dict[str, Dict[str, Any]] = {}

        for modality, items in ranked_lists.items():
            w = weights.get(modality, 1.0 / max(1, len(ranked_lists)))
            
            for rank, item in enumerate(items, start=1):
                item_id = str(item.get("id") or item.get("timestamp") or rank)
                
                if item_id not in fused_results:
                    fused_results[item_id] = {
                        "item_data": item,
                        "fused_score": 0.0,
                        "modality_scores": {}
                    }
                
                # RRF score addition
                rrf_increment = w / (self.k + rank)
                fused_results[item_id]["fused_score"] += rrf_increment
                fused_results[item_id]["modality_scores"][modality] = rrf_increment

        return fused_results
