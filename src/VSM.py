import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Tuple


class VSMModel:
    
    
    def __init__(self):
        # Vocabulaire et documents
        self.vocabulary = []      # Liste ordonnée des termes
        self.doc_ids = []         # Liste ordonnée des doc_ids
        
        # Matrice TF-IDF (M × N)
        self.doc_vectors = None   # Vecteurs TF-IDF des documents
        
        # Index inversé pour accès rapide
        self.inverted_index = {}  # {term: {doc_id: tfidf_weight}}
    
    
    def load_inverted_index(self, filepath: str, verbose: bool = True):
       
        if verbose:
            print("\n" + "="*80)
            print("CHARGEMENT DE L'INVERTED INDEX")
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
        
        # Construire la matrice TF-IDF (M × N)
        self.doc_vectors = np.zeros((M, N), dtype=np.float32)
        
        for i, term in enumerate(self.vocabulary):
            for j, doc_id in enumerate(self.doc_ids):
                if doc_id in term_doc_weights[term]:
                    self.doc_vectors[i, j] = term_doc_weights[term][doc_id]
        
        # Stocker l'index inversé
        self.inverted_index = {term: dict(term_doc_weights[term]) 
                              for term in self.vocabulary}
        
        if verbose:
            non_zero = np.count_nonzero(self.doc_vectors)
            density = (non_zero / (M * N)) * 100
            print(f"\n✅ Matrice document-terme construite: {M} × {N}")
            print(f"   - Éléments non-nuls: {non_zero:,}")
            print(f"   - Densité: {density:.2f}%")
            print("="*80)
    
    
    def create_query_vector(self, query_terms: List[str]) -> np.ndarray:
        
        # Créer le vecteur de requête (présence binaire)
        query_vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        
        found_terms = []
        for term in query_terms:
            if term in self.vocabulary:
                idx = self.vocabulary.index(term)
                query_vector[idx] = 1.0  # Pondération binaire
                found_terms.append(term)
        
        if len(found_terms) == 0:
            return None
        
        return query_vector
    
    
    def compute_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
       
        # Produit scalaire query · documents
        dot_products = query_vector @ self.doc_vectors  # (N,)
        
        # Norme de la requête
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(self.doc_ids))
        
        # Normes des documents
        doc_norms = np.linalg.norm(self.doc_vectors, axis=0)  # (N,)
        
        # Éviter division par zéro
        doc_norms[doc_norms == 0] = 1
        
        # Similarité cosinus
        similarities = dot_products / (query_norm * doc_norms)
        
        return similarities
    
    
    def rank_documents(self, query_terms: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        
        # Créer le vecteur de requête
        query_vector = self.create_query_vector(query_terms)
        
        if query_vector is None:
            # Aucun terme trouvé
            return []
        
        # Calculer les similarités cosinus
        similarities = self.compute_cosine_similarity(query_vector)
        
        # Créer la liste (doc_id, score)
        doc_scores = [(self.doc_ids[i], similarities[i]) 
                     for i in range(len(self.doc_ids))]
        
        # Trier par score décroissant
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k si spécifié
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    
    def fit(self, inverted_index_path: str, verbose: bool = True):
       
        if verbose:
            print("\n" + "="*80)
            print("INITIALISATION DU MODÈLE VSM")
            print("="*80)
            print("Mesure de similarité: Cosine")
        
        # Charger l'inverted index
        self.load_inverted_index(inverted_index_path, verbose)
        
        if verbose:
            print("\n" + "="*80)
            print("✅ MODÈLE VSM PRÊT")
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
    
    # Créer et charger le modèle
    print("="*80)
    print("TEST DU MODÈLE VSM SUR TOUTES LES REQUÊTES MED.QRY")
    print("="*80)
    
    vsm = VSMModel()
    vsm.fit(INVERTED_INDEX_PATH, verbose=True)
    
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
        results = vsm.search(query_terms, top_k=10, verbose=True)
        
        if not results:
            print("⚠️  Aucun résultat trouvé pour cette requête")
    
    print("\n" + "="*80)
    print("✅ TEST TERMINÉ SUR TOUTES LES REQUÊTES")
    print("="*80)