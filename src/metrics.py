
import math
from typing import List, Dict, Tuple
import numpy as np


class IRMetrics:
    """Classe pour calculer les métriques d'évaluation des systèmes de RI"""
    
    def __init__(self, relevance_judgments: Dict[int, List[int]]):
        """
        Args:
            relevance_judgments: Dict {query_id: [liste des doc_ids pertinents]}
        """
        self.relevance_judgments = relevance_judgments
    
    
    def precision(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        
        Precision = Nombre de documents pertinents sélectionnés / 
                    Nombre total de documents sélectionnés
        
        """
        if len(retrieved) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(retrieved)
    
    
    def recall(self, retrieved: List[int], relevant: List[int]) -> float:
        """
       
        
        Recall = Nombre de documents pertinents sélectionnés / 
                 Nombre total de documents pertinents
        
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
       
        F = 2 * P * R / (P + R)
    
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    
    def precision_recall_curve(self, ranked_list: List[int], relevant: List[int]) -> Tuple[List[float], List[float]]:
        """
       
        
        Pour chaque document pertinent dans la liste classée,
        on calcule la précision et le rappel à cette position.
    
        """
        recalls = []
        precisions = []
        relevant_set = set(relevant)
        num_relevant = len(relevant)
        
        if num_relevant == 0:
            return [], []
        
        relevant_retrieved = 0
        
        for i, doc_id in enumerate(ranked_list):
            if doc_id in relevant_set:
                relevant_retrieved += 1
                # Précision à la position i+1
                prec = relevant_retrieved / (i + 1)
                # Rappel à la position i+1
                rec = relevant_retrieved / num_relevant
                
                precisions.append(prec)
                recalls.append(rec)
        
        return recalls, precisions
    
    
    def interpolated_precision_recall_curve(self, recalls: List[float], precisions: List[float]) -> Tuple[List[float], List[float]]:
        """
        
        
        Pour chaque niveau de rappel standard (0.0, 0.1, 0.2, ..., 1.0),
        la précision interpolée est la précision maximale pour tous les 
        rappels >= rappel_standard
        
        P(rj) = max_{r >= rj} P(r)

        """
        standard_recalls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        interpolated_precisions = []
        
        # Ajouter le point (0, 0) si nécessaire
        if not recalls or recalls[0] > 0:
            recalls = [0.0] + list(recalls)
            precisions = [precisions[0] if precisions else 0.0] + list(precisions)
        
        for std_recall in standard_recalls:
            # Trouver toutes les précisions où recall >= std_recall
            max_prec = 0.0
            for r, p in zip(recalls, precisions):
                if r >= std_recall:
                    max_prec = max(max_prec, p)
            
            interpolated_precisions.append(max_prec)
        
        return standard_recalls, interpolated_precisions
    
    
    def average_precision(self, ranked_list: List[int], relevant: List[int]) -> float:
        """
        
        AP = (1/R) * Σ P(k) pour chaque position k où un document pertinent apparaît
        
        """
        relevant_set = set(relevant)
        num_relevant = len(relevant)
        
        if num_relevant == 0:
            return 0.0
        
        relevant_retrieved = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(ranked_list):
            if doc_id in relevant_set:
                relevant_retrieved += 1
                # Précision à cette position
                precision_at_k = relevant_retrieved / (i + 1)
                sum_precisions += precision_at_k
        
        return sum_precisions / num_relevant
    
    
    def mean_average_precision(self, results: Dict[int, List[int]]) -> float:
        """
        
        MAP = (1/|Q|) * Σ AP(q) pour toutes les requêtes q
        
        """
        if len(results) == 0:
            return 0.0
        
        sum_ap = 0.0
        
        for query_id, ranked_list in results.items():
            if query_id in self.relevance_judgments:
                relevant = self.relevance_judgments[query_id]
                ap = self.average_precision(ranked_list, relevant)
                sum_ap += ap
        
        return sum_ap / len(results)
    
    
    def interpolated_average_precision(self, ranked_list: List[int], relevant: List[int]) -> float:
        """
        
        Calcule l'AP en utilisant les précisions interpolées aux 11 points standards
        
        """
        recalls, precisions = self.precision_recall_curve(ranked_list, relevant)
        
        if not recalls:
            return 0.0
        
        _, interpolated_precs = self.interpolated_precision_recall_curve(recalls, precisions)
        
        # Moyenne des 11 points interpolés
        return sum(interpolated_precs) / len(interpolated_precs)
    
    
    def interpolated_map(self, results: Dict[int, List[int]]) -> float:
        
        if len(results) == 0:
            return 0.0
        
        sum_iap = 0.0
        
        for query_id, ranked_list in results.items():
            if query_id in self.relevance_judgments:
                relevant = self.relevance_judgments[query_id]
                iap = self.interpolated_average_precision(ranked_list, relevant)
                sum_iap += iap
        
        return sum_iap / len(results)
    
    
    def precision_at_k(self, ranked_list: List[int], relevant: List[int], k: int) -> float:
        """
       
        
        P@K = Nombre de documents pertinents dans les K premiers / K
        
        """
        if k == 0:
            return 0.0
        
        top_k = ranked_list[:k]
        relevant_set = set(relevant)
        
        relevant_in_top_k = len(set(top_k) & relevant_set)
        
        return relevant_in_top_k / k
    
    
    def r_precision(self, ranked_list: List[int], relevant: List[int]) -> float:
        """        
        C'est la précision calculée après avoir examiné les R premiers 
        documents, où R est le nombre total de documents pertinents.
        
        """
        r = len(relevant)
        return self.precision_at_k(ranked_list, relevant, r)
    
    
    def reciprocal_rank(self, ranked_list: List[int], relevant: List[int]) -> float:
        """
        
        RR = 1 / rang du premier document pertinent
        
        """
        relevant_set = set(relevant)
        
        for i, doc_id in enumerate(ranked_list):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    
    def dcg_at_k(self, ranked_list: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        """
      
        DCG@p = Σ (rel_i / log2(i)) pour i de 1 à p
        où rel_i est le score de pertinence du document à la position i
        
        Note: Le premier document (i=1) n'est pas divisé par log2(1)=0,
              donc on utilise rel_1 directement
        
        """
        dcg = 0.0
        
        for i, doc_id in enumerate(ranked_list[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            
            if i == 0:
                # Premier document: pas de discount
                dcg += rel
            else:
                # Documents suivants: discount par log2(i+1)
                dcg += rel / math.log2(i + 1)
        
        return dcg
    
    
    def idcg_at_k(self, relevance_scores: Dict[int, float], k: int) -> float:
        """
        
        C'est le DCG maximal possible, obtenu en classant les documents
        par ordre décroissant de pertinence.
    
        """
        # Trier les documents par score de pertinence décroissant
        sorted_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Créer la liste idéale
        ideal_list = [doc_id for doc_id, score in sorted_docs]
        
        return self.dcg_at_k(ideal_list, relevance_scores, k)
    
    
    def ndcg_at_k(self, ranked_list: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        """        
        nDCG@K = DCG@K / IDCG@K
        
        """
        dcg = self.dcg_at_k(ranked_list, relevance_scores, k)
        idcg = self.idcg_at_k(relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    
    def gain_percentage(self, value_a: float, value_b: float) -> float:
       
        if value_b == 0:
            return 0.0
        
        return ((value_a - value_b) / value_b) * 100
    
    
    def evaluate_query(self, query_id: int, ranked_list: List[int], 
                       relevance_scores: Dict[int, float] = None) -> Dict[str, float]:
       
        if query_id not in self.relevance_judgments:
            return {}
        
        relevant = self.relevance_judgments[query_id]
        
        # Précision et Rappel finaux
        prec = self.precision(ranked_list, relevant)
        rec = self.recall(ranked_list, relevant)
        
        # Courbes P-R
        recalls, precisions = self.precision_recall_curve(ranked_list, relevant)
        
        metrics = {
            'precision': prec,
            'recall': rec,
            'f1_score': self.f1_score(prec, rec),
            'average_precision': self.average_precision(ranked_list, relevant),
            'interpolated_ap': self.interpolated_average_precision(ranked_list, relevant),
            'p@5': self.precision_at_k(ranked_list, relevant, 5),
            'p@10': self.precision_at_k(ranked_list, relevant, 10),
            'r_precision': self.r_precision(ranked_list, relevant),
            'reciprocal_rank': self.reciprocal_rank(ranked_list, relevant),
            'pr_curve': (recalls, precisions)
        }
        
        # DCG et nDCG si les scores de pertinence sont fournis
        if relevance_scores:
            metrics['dcg@20'] = self.dcg_at_k(ranked_list, relevance_scores, 20)
            metrics['ndcg@20'] = self.ndcg_at_k(ranked_list, relevance_scores, 20)
        
        return metrics
    
    
    def evaluate_system(self, results: Dict[int, List[int]], 
                       relevance_scores_per_query: Dict[int, Dict[int, float]] = None) -> Dict[str, float]:
       
        all_metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'average_precision': [],
            'interpolated_ap': [],
            'p@5': [],
            'p@10': [],
            'r_precision': [],
            'reciprocal_rank': [],
            'dcg@20': [],
            'ndcg@20': []
        }
        
        for query_id, ranked_list in results.items():
            if query_id not in self.relevance_judgments:
                continue
            
            rel_scores = relevance_scores_per_query.get(query_id, None) if relevance_scores_per_query else None
            query_metrics = self.evaluate_query(query_id, ranked_list, rel_scores)
            
            for key in all_metrics.keys():
                if key in query_metrics and key != 'pr_curve':
                    all_metrics[key].append(query_metrics[key])
        
        # Calculer les moyennes
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        # Ajouter MAP
        avg_metrics['map'] = self.mean_average_precision(results)
        avg_metrics['interpolated_map'] = self.interpolated_map(results)
        
        return avg_metrics


