import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Tuple


class LSIModel:
  
    def __init__(self, k: int = 100):
        
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
                    doc_id = int(parts[1])  # Convertir en int pour tri
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
        
     
        self.S2_D = (self.Sk @ self.Sk) @ self.VTk
        
        if verbose:
            print(f"\n✅ Matrice de projection M: {self.M.shape}")
            print(f"✅ S² @ D précalculé: {self.S2_D.shape}")
            print("="*80)
    
    
    def project_query(self, query_terms: List[str]) -> np.ndarray:
        
        # Créer le vecteur de requête dans l'espace original
        q = np.zeros(len(self.vocabulary), dtype=np.float32)
        
        # Marquer les termes présents dans la requête (présence binaire 
        found_terms = []
        for term in query_terms:
            if term in self.vocabulary:
                idx = self.vocabulary.index(term)
                q[idx] = 1.0 
                found_terms.append(term)
        
        if len(found_terms) == 0:
            # Aucun terme de la requête n'existe dans le vocabulaire
            return None
        
        q_new = q.T @ self.M  # Shape: (k,)
        
        return q_new
    
    
    def calculate_similarity(self, q_new: np.ndarray) -> np.ndarray:
        
       
        sim = q_new @ self.S2_D
        
        return sim
    
    
    def rank_documents(self, query_terms: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        
        q_new = self.project_query(query_terms)
        
        if q_new is None:
            # Aucun terme trouvé, retourner liste vide
            return []
        
        similarities = self.calculate_similarity(q_new)
        
        # Créer la liste (doc_id, score)
        doc_scores = [(self.doc_ids[i], similarities[i]) for i in range(len(self.doc_ids))]
        
        # Trier par score décroissant
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k si spécifié
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    
    def fit(self, inverted_index_path: str, verbose: bool = True):
        
        if verbose:
            print("\n" + "="*80)
            print("ENTRAÎNEMENT DU MODÈLE LSI")
            print("="*80)
            print(f"Paramètre k: {self.k}")
            print(f"Formules utilisées: CODE 2")
        
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
    
    
    def search(self, query_terms: List[str], top_k: int = 10, verbose: bool = False) -> List[int]:
       
        doc_scores = self.rank_documents(query_terms, top_k=top_k)
        
        if verbose and doc_scores:
            print(f"\n🔍 Top {min(top_k, len(doc_scores))} documents:")
            print(f"{'Rang':<6} {'Doc ID':<10} {'Score':<12}")
            print("-" * 30)
            for rank, (doc_id, score) in enumerate(doc_scores[:top_k], 1):
                print(f"{rank:<6} {doc_id:<10} {score:.6f}")
        
        # Retourner seulement les doc_ids
        return [doc_id for doc_id, score in doc_scores]


# ============================================================================
# TEST SUR TOUTES LES REQUÊTES MED.QRY
# ============================================================================

if __name__ == "__main__":
    
    from medline_parser import parse_med_qry
    from preprocessing import MEDLINEPreprocessor
    
    # Chemins
    INVERTED_INDEX_PATH = r"C:\Users\pc\Desktop\RI_Project\data\output\inverted_index.txt"
    MED_QRY_PATH = r"C:\Users\pc\Desktop\RI_Project\data\MED.QRY"
    
    # Vérifier les fichiers
    if not os.path.exists(INVERTED_INDEX_PATH):
        print(f"❌ ERREUR: Fichier non trouvé: {INVERTED_INDEX_PATH}")
        print("\n💡 Assurez-vous d'avoir exécuté preprocessing.py pour générer l'inverted index")
        exit(1)
    
    if not os.path.exists(MED_QRY_PATH):
        print(f"❌ ERREUR: Fichier non trouvé: {MED_QRY_PATH}")
        exit(1)
    
    # Créer et entraîner le modèle
    print("="*80)
    print("TEST DU MODÈLE LSI SUR TOUTES LES REQUÊTES MED.QRY")
    print("="*80)
    
    lsi = LSIModel(k=100)
    lsi.fit(INVERTED_INDEX_PATH, verbose=True)
    
    # Charger les requêtes
    print("\n📄 Chargement des requêtes...")
    queries = parse_med_qry(MED_QRY_PATH)
    print(f"✅ {len(queries)} requêtes chargées")
    
    # Créer le preprocessor
    preprocessor = MEDLINEPreprocessor()
    
    # Tester chaque requête
    print("\n" + "="*80)
    print("TRAITEMENT DES REQUÊTES")
    print("="*80)
    
    for query in queries:
        query_id = query.query_id
        query_text = query.text
        
        # Preprocesser la requête
        query_terms = preprocessor.preprocess_text(query_text)
        
        print(f"\n{'='*80}")
        print(f"📝 Requête {query_id}")
        print(f"{'='*80}")
        print(f"Texte: {query_text[:100]}...")
        print(f"Termes preprocessés: {query_terms[:10]}...")
        
        # Rechercher les documents
        results = lsi.search(query_terms, top_k=10, verbose=True)
        
        if not results:
            print("⚠️  Aucun résultat trouvé pour cette requête")
    
    print("\n" + "="*80)
    print("✅ TEST TERMINÉ SUR TOUTES LES REQUÊTES")
    print("="*80)