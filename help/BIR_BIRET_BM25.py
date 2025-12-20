import numpy as np
import pandas as pd
from collections import defaultdict
import math
import nltk
from datetime import datetime
from typing import Dict, List, Tuple

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configuration
TFIDF_FILE = r"C:\Users\pc\Desktop\RI\DocTerm_Porter\DocTerm_Porter_FreqWeight.txt"

# Requêtes et documents pertinents
QUERIES = {
    'q1': {
        'text': 'large language models for information retrieval and ranking',
        'relevant': ['D2', 'D4']
    },
    'q2': {
        'text': 'LLM for information retrieval and Ranking',
        'relevant': ['D2', 'D4']
    },
    'q3': {
        'text': 'query Reformulation in information retrieval',
        'relevant': ['D4', 'D1']
    },
    'q4': {
        'text': 'ranking Documents',
        'relevant': ['D2', 'D1']
    },
    'q5': {
        'text': 'Optimizing recommendation systems with LLMs by leveraging item metadata',
        'relevant': ['D3', 'D6']
    }
}

class ProbabilisticIRModel:
    def __init__(self, file_path):
        """Initialise le modèle de RI probabiliste"""
        self.file_path = file_path
        self.term_doc_matrix = None
        self.binary_matrix = None
        self.term_freq_matrix = None
        self.documents = []
        self.terms = []
        self.N = 0
        self.doc_lengths = {}
        self.avg_doc_length = 0
        
    def load_data(self):
        """Charge les données du fichier TF-IDF"""
        print("📂 Chargement des données...")
        
        try:
            import os
            if not os.path.exists(self.file_path):
                print(f"❌ Le fichier n'existe pas: {self.file_path}")
                return False
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            term_doc_dict = defaultdict(dict)
            term_freq_dict = defaultdict(dict)
            all_docs = set()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 4:
                    term, doc, freq, weight = parts
                    try:
                        term_doc_dict[term][doc] = float(weight)
                        term_freq_dict[term][doc] = int(freq)
                        all_docs.add(doc)
                        if term not in self.terms:
                            self.terms.append(term)
                    except ValueError:
                        continue
            
            self.documents = sorted(list(all_docs))
            self.N = len(self.documents)
            
            # Matrice TF-IDF
            matrix_data = []
            for term in self.terms:
                row = [term_doc_dict[term].get(doc, 0.0) for doc in self.documents]
                matrix_data.append(row)
            
            self.term_doc_matrix = pd.DataFrame(matrix_data, index=self.terms, columns=self.documents)
            self.binary_matrix = (self.term_doc_matrix > 0).astype(int)
            
            # Matrice de fréquences
            freq_data = []
            for term in self.terms:
                row = [term_freq_dict[term].get(doc, 0) for doc in self.documents]
                freq_data.append(row)
            
            self.term_freq_matrix = pd.DataFrame(freq_data, index=self.terms, columns=self.documents)
            
            # Calculer longueurs de documents
            for doc in self.documents:
                self.doc_lengths[doc] = self.term_freq_matrix[doc].sum()
            
            self.avg_doc_length = np.mean(list(self.doc_lengths.values()))
            
            print(f"✅ {len(self.terms)} termes, {self.N} documents chargés")
            print(f"📏 Longueur moyenne des documents: {self.avg_doc_length:.2f}\n")
            return True
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_query(self, query_text):
        """Traite une requête avec stemming"""
        from nltk.stem import PorterStemmer
        import re
        
        stemmer = PorterStemmer()
        query_text = query_text.lower()
        query_text = re.sub(r'[^a-z\s]', ' ', query_text)
        tokens = query_text.split()
        stemmed = [stemmer.stem(t) for t in tokens if len(t) > 2]
        valid = [t for t in stemmed if t in self.binary_matrix.index]
        
        return valid
    
    def compute_ni(self, term):
        """Calcule ni: nombre de documents contenant le terme"""
        return self.binary_matrix.loc[term].sum()
    
    def get_term_freq(self, term, doc):
        """Retourne la fréquence du terme dans le document"""
        if term in self.term_freq_matrix.index and doc in self.term_freq_matrix.columns:
            return self.term_freq_matrix.loc[term, doc]
        return 0
    
    def get_tfidf_weight(self, term, doc):
        """Retourne le poids TF-IDF du terme dans le document"""
        if term in self.term_doc_matrix.index and doc in self.term_doc_matrix.columns:
            return self.term_doc_matrix.loc[term, doc]
        return 0.0
    
    # ========== CLASSIC BIR ==========
    
    def bir_without_learning(self, query_id):
        """BIR Classique SANS données d'apprentissage"""
        query_info = QUERIES[query_id]
        valid_terms = self.process_query(query_info['text'])
        
        if not valid_terms:
            return {}
        
        rsv_scores = {}
        for doc in self.documents:
            rsv = 0
            for term in valid_terms:
                if self.binary_matrix.loc[term, doc] == 1:
                    ni = self.compute_ni(term)
                    rsv += math.log10((self.N - ni + 0.5) / (ni + 0.5))
            rsv_scores[doc] = rsv
        
        return sorted(rsv_scores.items(), key=lambda x: x[1], reverse=True)
    
    def bir_with_learning(self, query_id):
        """BIR Classique AVEC données d'apprentissage"""
        query_info = QUERIES[query_id]
        valid_terms = self.process_query(query_info['text'])
        relevant_docs = query_info['relevant']
        
        if not valid_terms:
            return {}, 0, []
        
        R = len(relevant_docs)
        rsv_scores = {}
        
        for doc in self.documents:
            rsv = 0
            for term in valid_terms:
                if self.binary_matrix.loc[term, doc] == 1:
                    n = self.compute_ni(term)
                    r = sum(1 for rel_doc in relevant_docs 
                           if rel_doc in self.documents and 
                           self.binary_matrix.loc[term, rel_doc] == 1)
                    
                    numerator = (r + 0.5) / (R - r + 0.5)
                    denominator = (n - r + 0.5) / (self.N - R - n + r + 0.5)
                    rsv += math.log10(numerator / denominator)
            
            rsv_scores[doc] = rsv
        
        ranking = sorted(rsv_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculer précision
        top_R = [doc for doc, _ in ranking[:R]]
        relevant_in_top = set(top_R) & set(relevant_docs)
        precision = len(relevant_in_top) / R if R > 0 else 0
        
        return ranking, precision, relevant_docs
    
    def extended_bir_without_learning(self, query_id):
        """Extended BIR SANS données d'apprentissage"""
        query_info = QUERIES[query_id]
        valid_terms = self.process_query(query_info['text'])
        
        if not valid_terms:
            return {}
        
        rsv_scores = {}
        
        for doc in self.documents:
            rsv = 0
            for term in valid_terms:
                w_ij = self.get_tfidf_weight(term, doc)
                qtf_i = 1
                ni = self.compute_ni(term)
                idf_prob = math.log10((self.N - ni + 0.5) / (ni + 0.5))
                rsv += w_ij * qtf_i * idf_prob
            
            rsv_scores[doc] = rsv
        
        return sorted(rsv_scores.items(), key=lambda x: x[1], reverse=True)
    
    def extended_bir_with_learning(self, query_id):
        """Extended BIR AVEC données d'apprentissage"""
        query_info = QUERIES[query_id]
        valid_terms = self.process_query(query_info['text'])
        relevant_docs = query_info['relevant']
        
        if not valid_terms:
            return {}, 0, []
        
        R = len(relevant_docs)
        rsv_scores = {}
        
        for doc in self.documents:
            rsv = 0
            for term in valid_terms:
                w_ij = self.get_tfidf_weight(term, doc)
                qtf_i = 1
                n = self.compute_ni(term)
                r = sum(1 for rel_doc in relevant_docs 
                       if rel_doc in self.documents and 
                       self.binary_matrix.loc[term, rel_doc] == 1)
                
                numerator = (r + 0.5) * (self.N - R - n + r + 0.5)
                denominator = (n - r + 0.5) * (R - r + 0.5)
                prob_weight = math.log10(numerator / denominator)
                rsv += w_ij * qtf_i * prob_weight
            
            rsv_scores[doc] = rsv
        
        ranking = sorted(rsv_scores.items(), key=lambda x: x[1], reverse=True)
        top_R = [doc for doc, _ in ranking[:R]]
        relevant_in_top = set(top_R) & set(relevant_docs)
        precision = len(relevant_in_top) / R if R > 0 else 0
        
        return ranking, precision, relevant_docs
    
    def bm25(self, query_id, k1=1.5, b=0.75):
        """Modèle BM25"""
        query_info = QUERIES[query_id]
        valid_terms = self.process_query(query_info['text'])
        relevant_docs = query_info['relevant']
        
        if not valid_terms:
            return {}, 0, []
        
        rsv_scores = {}
        
        for doc in self.documents:
            rsv = 0
            dl = self.doc_lengths[doc]
            
            for term in valid_terms:
                tf = self.get_term_freq(term, doc)
                if tf > 0:
                    ni = self.compute_ni(term)
                    idf = math.log10((self.N - ni + 0.5) / (ni + 0.5))
                    normalization = 1 - b + b * (dl / self.avg_doc_length)
                    tf_component = (tf * (k1 + 1)) / (tf + k1 * normalization)
                    rsv += idf * tf_component
            
            rsv_scores[doc] = rsv
        
        ranking = sorted(rsv_scores.items(), key=lambda x: x[1], reverse=True)
        R = len(relevant_docs)
        top_R = [doc for doc, _ in ranking[:R]]
        relevant_in_top = set(top_R) & set(relevant_docs)
        precision = len(relevant_in_top) / R if R > 0 else 0
        
        return ranking, precision, relevant_docs
    
    # ========== AFFICHAGE PAR MODÈLE ==========
    
    def display_bir_without_learning(self):
        """Affiche uniquement les résultats du BIR sans apprentissage"""
        print("\n" + "="*100)
        print(" MODÈLE 1: BIR CLASSIQUE SANS APPRENTISSAGE ".center(100, "="))
        print("="*100)
        print("Formule: RSV(q,d) = Σ log10((N - n_i + 0.5) / (n_i + 0.5)) pour chaque terme présent")
        print("="*100 + "\n")
        
        for qid in ['q1', 'q2', 'q3', 'q4', 'q5']:
            query_info = QUERIES[qid]
            valid_terms = self.process_query(query_info['text'])
            ranking = self.bir_without_learning(qid)
            
            print(f"📌 REQUÊTE {qid.upper()}")
            print(f"   Texte: {query_info['text']}")
            print(f"   Termes: {', '.join(valid_terms)}")
            print(f"   Documents pertinents: {', '.join(query_info['relevant'])}")
            print(f"\n   {'Rang':<6} {'Document':<12} {'Score RSV':<15} {'Statut'}")
            print(f"   {'-'*50}")
            
            for rank, (doc, score) in enumerate(ranking, 1):
                status = "✓ PERTINENT" if doc in query_info['relevant'] else ""
                print(f"   {rank:<6} {doc:<12} {score:>12.6f}   {status}")
            
            print("\n")
    
    def display_bir_with_learning(self):
        """Affiche uniquement les résultats du BIR avec apprentissage"""
        print("\n" + "="*100)
        print(" MODÈLE 2: BIR CLASSIQUE AVEC APPRENTISSAGE ".center(100, "="))
        print("="*100)
        print("Formule: RSV(q,d) = Σ log10((r+0.5)(N-R-n+r+0.5) / ((n-r+0.5)(R-r+0.5)))")
        print("="*100 + "\n")
        
        precisions = []
        
        for qid in ['q1', 'q2', 'q3', 'q4', 'q5']:
            query_info = QUERIES[qid]
            valid_terms = self.process_query(query_info['text'])
            ranking, precision, relevant_docs = self.bir_with_learning(qid)
            precisions.append(precision)
            
            print(f"📌 REQUÊTE {qid.upper()}")
            print(f"   Texte: {query_info['text']}")
            print(f"   Termes: {', '.join(valid_terms)}")
            print(f"   Documents pertinents: {', '.join(relevant_docs)}")
            print(f"   R = {len(relevant_docs)} documents pertinents")
            print(f"\n   {'Rang':<6} {'Document':<12} {'Score RSV':<15} {'Statut'}")
            print(f"   {'-'*50}")
            
            for rank, (doc, score) in enumerate(ranking, 1):
                status = "✓ PERTINENT" if doc in relevant_docs else ""
                print(f"   {rank:<6} {doc:<12} {score:>12.6f}   {status}")
            
            print(f"\n   🎯 Précision@{len(relevant_docs)}: {precision*100:.1f}%")
            print("\n")
        
        print(f"📊 PRÉCISION MOYENNE: {np.mean(precisions)*100:.1f}%\n")
    
    def display_extended_bir_without_learning(self):
        """Affiche uniquement les résultats de l'Extended BIR sans apprentissage"""
        print("\n" + "="*100)
        print(" MODÈLE 3: EXTENDED BIR SANS APPRENTISSAGE ".center(100, "="))
        print("="*100)
        print("Formule: RSV(q,d) = Σ w_ij · qtf_i · log10((N - n_i + 0.5) / (n_i + 0.5))")
        print("="*100 + "\n")
        
        for qid in ['q1', 'q2', 'q3', 'q4', 'q5']:
            query_info = QUERIES[qid]
            valid_terms = self.process_query(query_info['text'])
            ranking = self.extended_bir_without_learning(qid)
            
            print(f"📌 REQUÊTE {qid.upper()}")
            print(f"   Texte: {query_info['text']}")
            print(f"   Termes: {', '.join(valid_terms)}")
            print(f"   Documents pertinents: {', '.join(query_info['relevant'])}")
            print(f"\n   {'Rang':<6} {'Document':<12} {'Score RSV':<15} {'Statut'}")
            print(f"   {'-'*50}")
            
            for rank, (doc, score) in enumerate(ranking, 1):
                status = "✓ PERTINENT" if doc in query_info['relevant'] else ""
                print(f"   {rank:<6} {doc:<12} {score:>12.6f}   {status}")
            
            print("\n")
    
    def display_extended_bir_with_learning(self):
        """Affiche uniquement les résultats de l'Extended BIR avec apprentissage"""
        print("\n" + "="*100)
        print(" MODÈLE 4: EXTENDED BIR AVEC APPRENTISSAGE ".center(100, "="))
        print("="*100)
        print("Formule: RSV(q,d) = Σ w_ij · qtf_i · log10((r+0.5)(N-R-n+r+0.5) / ((n-r+0.5)(R-r+0.5)))")
        print("="*100 + "\n")
        
        precisions = []
        
        for qid in ['q1', 'q2', 'q3', 'q4', 'q5']:
            query_info = QUERIES[qid]
            valid_terms = self.process_query(query_info['text'])
            ranking, precision, relevant_docs = self.extended_bir_with_learning(qid)
            precisions.append(precision)
            
            print(f"📌 REQUÊTE {qid.upper()}")
            print(f"   Texte: {query_info['text']}")
            print(f"   Termes: {', '.join(valid_terms)}")
            print(f"   Documents pertinents: {', '.join(relevant_docs)}")
            print(f"   R = {len(relevant_docs)} documents pertinents")
            print(f"\n   {'Rang':<6} {'Document':<12} {'Score RSV':<15} {'Statut'}")
            print(f"   {'-'*50}")
            
            for rank, (doc, score) in enumerate(ranking, 1):
                status = "✓ PERTINENT" if doc in relevant_docs else ""
                print(f"   {rank:<6} {doc:<12} {score:>12.6f}   {status}")
            
            print(f"\n   🎯 Précision@{len(relevant_docs)}: {precision*100:.1f}%")
            print("\n")
        
        print(f"📊 PRÉCISION MOYENNE: {np.mean(precisions)*100:.1f}%\n")
    
    def display_bm25(self):
        """Affiche uniquement les résultats du modèle BM25"""
        print("\n" + "="*100)
        print(" MODÈLE 5: BM25 ".center(100, "="))
        print("="*100)
        print("Formule: BM25(d,q) = Σ IDF(t_i) · (tf(t_i,d) · (k1+1)) / (tf(t_i,d) + k1·(1-b+b·dl/avdl))")
        print(f"Paramètres: k1=1.5, b=0.75, longueur moyenne={self.avg_doc_length:.2f}")
        print("="*100 + "\n")
        
        precisions = []
        
        for qid in ['q1', 'q2', 'q3', 'q4', 'q5']:
            query_info = QUERIES[qid]
            valid_terms = self.process_query(query_info['text'])
            ranking, precision, relevant_docs = self.bm25(qid)
            precisions.append(precision)
            
            print(f"📌 REQUÊTE {qid.upper()}")
            print(f"   Texte: {query_info['text']}")
            print(f"   Termes: {', '.join(valid_terms)}")
            print(f"   Documents pertinents: {', '.join(relevant_docs)}")
            print(f"   R = {len(relevant_docs)} documents pertinents")
            print(f"\n   {'Rang':<6} {'Document':<12} {'Score BM25':<15} {'Statut'}")
            print(f"   {'-'*50}")
            
            for rank, (doc, score) in enumerate(ranking, 1):
                status = "✓ PERTINENT" if doc in relevant_docs else ""
                print(f"   {rank:<6} {doc:<12} {score:>12.6f}   {status}")
            
            print(f"\n   🎯 Précision@{len(relevant_docs)}: {precision*100:.1f}%")
            print("\n")
        
        print(f"📊 PRÉCISION MOYENNE: {np.mean(precisions)*100:.1f}%\n")
    
    def run_all_models_separately(self):
        """Exécute tous les modèles avec affichage séparé"""
        print("\n" + "="*100)
        print(" AFFICHAGE SÉPARÉ DES MODÈLES PROBABILISTES ".center(100, "="))
        print("="*100)
        print(f"📊 Corpus: {self.N} documents, {len(self.terms)} termes")
        print(f"📏 Longueur moyenne: {self.avg_doc_length:.2f} termes/document")
        print("="*100)
        
        # Afficher chaque modèle séparément
        self.display_bir_without_learning()
        input("\nAppuyez sur Entrée pour voir le modèle suivant...")
        
        self.display_bir_with_learning()
        input("\nAppuyez sur Entrée pour voir le modèle suivant...")
        
        self.display_extended_bir_without_learning()
        input("\nAppuyez sur Entrée pour voir le modèle suivant...")
        
        self.display_extended_bir_with_learning()
        input("\nAppuyez sur Entrée pour voir le modèle suivant...")
        
        self.display_bm25()
        
        print("\n" + "="*100)
        print(" RÉSUMÉ COMPARATIF FINAL ".center(100, "="))
        print("="*100)
        
        # Tableau récapitulatif
        print(f"\n{'Query':<8} {'BIR+L':<12} {'Ext-BIR+L':<12} {'BM25':<12} {'Meilleur'}")
        print("-"*60)
        
        for qid in ['q1', 'q2', 'q3', 'q4', 'q5']:
            _, bir_prec, _ = self.bir_with_learning(qid)
            _, ext_prec, _ = self.extended_bir_with_learning(qid)
            _, bm25_prec, _ = self.bm25(qid)
            
            best = max([('BIR+L', bir_prec), ('Ext-BIR+L', ext_prec), ('BM25', bm25_prec)], 
                      key=lambda x: x[1])[0]
            
            print(f"{qid.upper():<8} {bir_prec*100:>8.1f}%   {ext_prec*100:>8.1f}%   "
                  f"{bm25_prec*100:>8.1f}%   {best}")
        
        print("="*100 + "\n")


# Programme principal
if __name__ == "__main__":
    try:
        print("🚀 Initialisation des modèles probabilistes de RI...\n")
        
        model = ProbabilisticIRModel(TFIDF_FILE)
        
        if model.load_data():
            model.run_all_models_separately()
            print("✅ Traitement terminé avec succès!")
        else:
            print("❌ Échec du chargement des données")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()