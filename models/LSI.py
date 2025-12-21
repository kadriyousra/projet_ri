import numpy as np
from collections import defaultdict
import os
import sys
from typing import List, Dict, Tuple


class LSIModel:
    """Latent Semantic Indexing (LSI) Model"""
    
    def __init__(self, k: int = 100):
        """
        Args:
            k: Nombre de dimensions latentes (réduction de dimensionnalité)
        """
        self.k = k
        
        # Vocabulaire et documents
        self.vocabulary = []      # Liste ordonnée des termes
        self.doc_ids = []         # Liste ordonnée des doc_ids
        
        # Matrices SVD
        self.W = None             # Matrice TF-IDF (M × N)
        self.U = None             # Matrice des termes (M × min(M,N))
        self.S = None             # Valeurs singulières (diagonale)
        self.VT = None            # Matrice des documents transposée (min(M,N) × N)
        
        # Matrices réduites (k dimensions)
        self.Uk = None            # U tronquée (M × k)
        self.Sk = None            # S tronquée (k × k)
        self.VTk = None           # VT tronquée (k × N)
        
        # Matrice de projection des requêtes
        self.M = None             # M = Uk @ Sk^-1 (M × k)
        
        # Cache pour S² @ D (formule exacte du code 2)
        self.S2_D = None          # (Sk @ Sk) @ VTk
    
    
    def load_inverted_index(self, filepath: str, verbose: bool = True):
        """Charge l'inverted index et construit la matrice TF-IDF"""
        if verbose:
            print("\n" + "="*80)
            print("ÉTAPE 1: CHARGEMENT DE L'INVERTED INDEX")
            print("="*80)
        
        # Structure: {term: {doc_id: weight}}
        term_doc_weights = defaultdict(dict)
        all_docs = set()
        all_terms = set()
        
        # Lire le fichier
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    term = parts[0]
                    doc_id = int(parts[1])
                    weight = float(parts[3])  # TF-IDF weight
                    
                    term_doc_weights[term][doc_id] = weight
                    all_terms.add(term)
                    all_docs.add(doc_id)
        
        # Créer les listes ordonnées
        self.vocabulary = sorted(list(all_terms))
        self.doc_ids = sorted(list(all_docs))
        
        M = len(self.vocabulary)  # Nombre de termes
        N = len(self.doc_ids)     # Nombre de documents
        
        if verbose:
            print(f"\n📊 Statistiques:")
            print(f"   - Termes dans le vocabulaire: {M}")
            print(f"   - Documents: {N}")
        
        # Construire la matrice TF-IDF W (M × N)
        self.W = np.zeros((M, N), dtype=np.float32)
        
        for i, term in enumerate(self.vocabulary):
            for j, doc_id in enumerate(self.doc_ids):
                if doc_id in term_doc_weights[term]:
                    self.W[i, j] = term_doc_weights[term][doc_id]
        
        if verbose:
            non_zero = np.count_nonzero(self.W)
            density = (non_zero / (M * N)) * 100
            print(f"\n✅ Matrice W construite: {M} × {N}")
            print(f"   - Éléments non-nuls: {non_zero:,}")
            print(f"   - Densité: {density:.2f}%")
            print("="*80)
    
    
    def apply_svd(self, verbose: bool = True):
        """Applique la décomposition SVD sur la matrice TF-IDF"""
        if verbose:
            print("\n" + "="*80)
            print("ÉTAPE 2: DÉCOMPOSITION SVD")
            print("="*80)
        
        self.U, s, self.VT = np.linalg.svd(self.W, full_matrices=False)
        self.S = np.diag(s)
        
        if verbose:
            print(f"\n📐 Dimensions des matrices:")
            print(f"   - U:  {self.U.shape}")
            print(f"   - S:  {self.S.shape}")
            print(f"   - VT: {self.VT.shape}")
            print(f"\n📊 Top 10 valeurs singulières:")
            for i, val in enumerate(s[:10], 1):
                print(f"   σ{i}: {val:.4f}")
            
            # Variance expliquée
            total_variance = np.sum(s**2)
            cumsum = np.cumsum(s**2)
            variance_k = (cumsum[self.k-1] / total_variance) * 100
            print(f"\n💡 Variance expliquée par k={self.k}: {variance_k:.2f}%")
            print("="*80)
    
    
    def reduce_dimensionality(self, verbose: bool = True):
        """Réduit la dimensionnalité à k dimensions"""
        if verbose:
            print("\n" + "="*80)
            print(f"ÉTAPE 3: RÉDUCTION À k={self.k} DIMENSIONS")
            print("="*80)
        
        # Tronquer aux k premières dimensions
        self.Uk = self.U[:, :self.k]
        self.Sk = self.S[:self.k, :self.k]
        self.VTk = self.VT[:self.k, :]
        
        if verbose:
            print(f"\n📐 Matrices réduites:")
            print(f"   - Uk:  {self.Uk.shape}")
            print(f"   - Sk:  {self.Sk.shape}")
            print(f"   - VTk: {self.VTk.shape}")
        
        # Calculer la matrice de projection M = Uk @ Sk^-1
        Sk_inv = np.linalg.inv(self.Sk)
        self.M = self.Uk @ Sk_inv
        
        # Précalculer S² @ D pour la similarité
        self.S2_D = (self.Sk @ self.Sk) @ self.VTk
        
        if verbose:
            print(f"\n✅ Matrice de projection M: {self.M.shape}")
            print(f"✅ S² @ D précalculé: {self.S2_D.shape}")
            print("="*80)
    
    
    def project_query(self, query_terms: List[str]) -> np.ndarray:
        """Projette une requête dans l'espace latent réduit"""
        # Créer le vecteur de requête dans l'espace original
        q = np.zeros(len(self.vocabulary), dtype=np.float32)
        
        # Marquer les termes présents dans la requête (présence binaire)
        found_terms = []
        for term in query_terms:
            if term in self.vocabulary:
                idx = self.vocabulary.index(term)
                q[idx] = 1.0
                found_terms.append(term)
        
        if len(found_terms) == 0:
            # Aucun terme de la requête n'existe dans le vocabulaire
            return None
        
        # Projection: q_new = q^T @ M
        q_new = q.T @ self.M  # Shape: (k,)
        
        return q_new
    
    
    def calculate_similarity(self, q_new: np.ndarray) -> np.ndarray:
        """Calcule les similarités entre la requête projetée et tous les documents"""
        # Similarité: sim = q_new @ (S² @ D)
        sim = q_new @ self.S2_D
        
        return sim
    
    
    def rank_documents(self, query_terms: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Classe tous les documents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            top_k: Si spécifié, limite le nombre de résultats (pour affichage)
                   Si None, retourne TOUS les documents (requis pour évaluation)
        
        Returns:
            Liste de tuples (doc_id, score) triée par score décroissant
        """
        # Projeter la requête
        q_new = self.project_query(query_terms)
        
        if q_new is None:
            # Aucun terme trouvé, retourner tous les docs avec score 0
            return [(doc_id, 0.0) for doc_id in self.doc_ids]
        
        # Calculer les similarités
        similarities = self.calculate_similarity(q_new)
        
        # Créer la liste (doc_id, score)
        doc_scores = [(self.doc_ids[i], float(similarities[i])) 
                     for i in range(len(self.doc_ids))]
        
        # Trier par score décroissant
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k si spécifié (pour affichage uniquement)
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    
    def fit(self, inverted_index_path: str, verbose: bool = True):
        """
        Entraîne le modèle LSI
        
        Args:
            inverted_index_path: Chemin vers l'inverted index
            verbose: Afficher les détails
        """
        if verbose:
            print("\n" + "="*80)
            print("ENTRAÎNEMENT DU MODÈLE LSI")
            print("="*80)
            print(f"Paramètre k: {self.k}")
        
        # Étape 1: Charger l'inverted index
        self.load_inverted_index(inverted_index_path, verbose)
        
        # Étape 2: Appliquer SVD
        self.apply_svd(verbose)
        
        # Étape 3: Réduire la dimensionnalité
        self.reduce_dimensionality(verbose)
        
        if verbose:
            print("\n" + "="*80)
            print("✅ MODÈLE LSI PRÊT")
            print("="*80)
    
    
    def search(self, query_terms: List[str], top_k: int = 10, 
              verbose: bool = False, return_all: bool = False) -> List[int]:
        """
        Recherche les documents pertinents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            top_k: Nombre de documents à retourner (ignoré si return_all=True)
            verbose: Afficher les résultats
            return_all: Si True, retourne TOUS les documents (pour évaluation)
        
        Returns:
            Liste des doc_ids classés par pertinence
        """
        # Obtenir tous les documents classés
        doc_scores = self.rank_documents(query_terms, top_k=None)
        
        if verbose and doc_scores:
            display_k = min(top_k, len(doc_scores))
            print(f"\n🔍 Top {display_k} documents:")
            print(f"{'Rang':<6} {'Doc ID':<10} {'Score':<12}")
            print("-" * 30)
            for rank, (doc_id, score) in enumerate(doc_scores[:display_k], 1):
                print(f"{rank:<6} {doc_id:<10} {score:.6f}")
        
        # Extraire les doc_ids
        ranked_list = [doc_id for doc_id, score in doc_scores]
        
        # Limiter seulement si demandé ET pas return_all
        if top_k and not return_all:
            ranked_list = ranked_list[:top_k]
        
        return ranked_list


# ============================================================================
# ÉVALUATION COMPLÈTE AVEC METRICS.PY
# ============================================================================

if __name__ == "__main__":
    
    # Ajouter le dossier src et evaluation au path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    src_dir = os.path.join(project_dir, 'src')
    eval_dir = os.path.join(project_dir, 'evaluation')
    sys.path.insert(0, src_dir)
    sys.path.insert(0, eval_dir)
    
    from medline_parser import parse_med_qry, parse_med_rel
    from preprocessing import MEDLINEPreprocessor
    from metrics import IRMetrics
    
    # Chemins
    INVERTED_INDEX_PATH = os.path.join(project_dir, "data", "ouput", "inverted_index.txt")
    MED_QRY_PATH = os.path.join(project_dir, "data", "MED.QRY")
    MED_REL_PATH = os.path.join(project_dir, "data", "MED.REL")
    
    # Vérifier les fichiers
    for path, name in [(INVERTED_INDEX_PATH, "Inverted Index"),
                       (MED_QRY_PATH, "MED.QRY"),
                       (MED_REL_PATH, "MED.REL")]:
        if not os.path.exists(path):
            print(f"❌ ERREUR: Fichier non trouvé: {path}")
            exit(1)
    
    print("="*80)
    print("ÉVALUATION COMPLÈTE DU MODÈLE LSI")
    print("="*80)
    
    # 1. Créer et entraîner le modèle
    print("\n📚 Étape 1: Entraînement du modèle LSI")
    lsi = LSIModel(k=100)
    lsi.fit(INVERTED_INDEX_PATH, verbose=True)
    
    # 2. Charger les données
    print("\n📄 Étape 2: Chargement des données")
    queries = parse_med_qry(MED_QRY_PATH)
    relevance_judgments = parse_med_rel(MED_REL_PATH)
    print(f"✅ {len(queries)} requêtes chargées")
    print(f"✅ {len(relevance_judgments)} jugements de pertinence chargés")
    
    # 3. Créer le preprocessor
    preprocessor = MEDLINEPreprocessor()
    
    # 4. Initialiser le système de métriques
    print("\n📊 Étape 3: Initialisation du système d'évaluation")
    metrics = IRMetrics(relevance_judgments, model_name="LSI_k100")
    
    # 5. Collecter tous les résultats
    print("\n🔍 Étape 4: Traitement de toutes les requêtes")
    results_per_query = {}
    relevance_scores_per_query = {}
    
    for query in queries:
        query_id = query.query_id
        query_text = query.text
        
        # Preprocesser la requête
        query_terms = preprocessor.preprocess_text(query_text)
        
        # ✅ CRITICAL: Obtenir TOUS les documents classés (pas de limitation top_k)
        doc_scores = lsi.rank_documents(query_terms, top_k=None)
        
        # Extraire les doc_ids et les scores
        ranked_list = [doc_id for doc_id, score in doc_scores]
        scores_dict = {doc_id: score for doc_id, score in doc_scores}
        
        results_per_query[query_id] = ranked_list
        relevance_scores_per_query[query_id] = scores_dict
        
        print(f"   Requête {query_id}: {len(ranked_list)} documents classés")
    
    # 6. Évaluer le système complet
    print("\n📈 Étape 5: Évaluation complète du système")
    all_results = metrics.evaluate_all_queries(
        results_per_query=results_per_query,
        relevance_scores_per_query=relevance_scores_per_query,  # Pour DCG/nDCG
        plot_curves=True,
        save_results=True,
        verbose=False  # Mettre True pour voir les détails de chaque requête
    )
    
    print("\n" + "="*80)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*80)
    print(f"📁 Résultats sauvegardés:")
    print(f"   - results/LSI_k100_results.txt")
    print(f"   - results/figures/LSI_k100/")
    print("="*80)