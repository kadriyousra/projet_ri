import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class BM25Model:
    """Modèle BM25 pour MEDLINE avec métriques complètes"""
    
    def __init__(self, inverted_index_path, doc_term_matrix_path, documents, queries, relevance_judgments):
        self.documents = documents
        self.queries = queries
        self.relevance_judgments = relevance_judgments
        
        # Charger l'index inversé
        self.inverted_index = self._load_inverted_index(inverted_index_path)
        
        # Charger les longueurs des documents
        self.doc_lengths = self._load_doc_lengths(doc_term_matrix_path)
        
        # Calculer les statistiques
        self.N = len(documents)
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.N if self.N > 0 else 0
        
        # Document IDs
        self.doc_ids = [doc.doc_id for doc in documents]
        
        # Créer les dossiers de résultats
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Crée la structure de dossiers pour les résultats"""
        base_dir = "results/figures/BM25"
        self.dirs = {
            'base': base_dir,
            'curves_non_interpolated': os.path.join(base_dir, "curves_non_interpolated"),
            'curves_interpolated': os.path.join(base_dir, "curves_interpolated"),
            'all_queries': base_dir
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_inverted_index(self, filepath):
        inverted_index = defaultdict(dict)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    term = parts[0]
                    doc_id = int(parts[1])
                    freq = int(parts[2])
                    inverted_index[term][doc_id] = freq
        return inverted_index
    
    def _load_doc_lengths(self, filepath):
        doc_lengths = defaultdict(int)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    doc_id = int(parts[0])
                    freq = int(parts[2])
                    doc_lengths[doc_id] += freq
        return doc_lengths
    
    def process_query(self, query_text):
        return query_text.lower().split()
    
    def get_term_freq(self, term, doc_id):
        return self.inverted_index.get(term, {}).get(doc_id, 0)
    
    def compute_ni(self, term):
        return len(self.inverted_index.get(term, {}))
    
    def calculate_precision_at_k(self, ranking, relevant_docs, k_values):
        """Calcule P@K pour différentes valeurs de K"""
        p_at_k = {}
        for k in k_values:
            if k > len(ranking):
                k = len(ranking)
            top_k = [doc_id for doc_id, _ in ranking[:k]]
            relevant_in_top_k = len([doc for doc in top_k if doc in relevant_docs])
            p_at_k[k] = relevant_in_top_k / k if k > 0 else 0.0
        return p_at_k
    
    def calculate_r_precision(self, ranking, relevant_docs):
        """Calcule R-Precision où R = nombre de documents pertinents"""
        R = len(relevant_docs)
        if R == 0:
            return 0.0
        
        top_R = [doc_id for doc_id, _ in ranking[:R]]
        relevant_in_top_R = len([doc for doc in top_R if doc in relevant_docs])
        return relevant_in_top_R / R
    
    def calculate_reciprocal_rank(self, ranking, relevant_docs):
        """Calcule le Reciprocal Rank (1/position du 1er doc pertinent)"""
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        return 0.0
    
    def calculate_dcg_at_k(self, ranking, relevant_docs, rsv_scores, k=20):
        """Calcule DCG@K avec les scores RSV comme degrés de pertinence"""
        dcg = 0.0
        for i in range(min(k, len(ranking))):
            doc_id, _ = ranking[i]
            # Utiliser RSV comme degré de pertinence si le doc est pertinent
            rel_i = rsv_scores.get(doc_id, 0.0) if doc_id in relevant_docs else 0.0
            
            if i == 0:
                dcg += rel_i
            else:
                dcg += rel_i / math.log2(i + 1)
        
        return dcg
    
    def calculate_idcg_at_k(self, relevant_docs, rsv_scores, k=20):
        """Calcule IDCG@K (DCG idéal)"""
        # Trier les docs pertinents par RSV décroissant
        relevant_with_scores = [(doc_id, rsv_scores.get(doc_id, 0.0)) 
                                for doc_id in relevant_docs]
        relevant_sorted = sorted(relevant_with_scores, key=lambda x: x[1], reverse=True)
        
        idcg = 0.0
        for i in range(min(k, len(relevant_sorted))):
            doc_id, rel_i = relevant_sorted[i]
            if i == 0:
                idcg += rel_i
            else:
                idcg += rel_i / math.log2(i + 1)
        
        return idcg
    
    def calculate_ndcg_at_k(self, ranking, relevant_docs, rsv_scores, k=20):
        """Calcule nDCG@K"""
        dcg = self.calculate_dcg_at_k(ranking, relevant_docs, rsv_scores, k)
        idcg = self.calculate_idcg_at_k(relevant_docs, rsv_scores, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def calculate_average_precision(self, precision_recall_points):
        """Calcule la Average Precision (AP)"""
        precisions = [p['precision'] for p in precision_recall_points if p['is_relevant']]
        if not precisions:
            return 0.0
        return sum(precisions) / len(precisions)
    
    def calculate_interpolated_average_precision(self, interpolated_curve):
        """Calcule la Average Precision Interpolée"""
        precisions = [p[1] for p in interpolated_curve]
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def calculate_f_measure(self, precision, recall):
        """Calcule la F-Mesure (F1-Score)"""
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)
    
    def interpolate_precision_recall(self, precision_recall_points):
        """Interpole la courbe Précision-Rappel aux points standards"""
        relevant_points = [(p['recall'], p['precision']) for p in precision_recall_points if p['is_relevant']]
        
        if not relevant_points:
            return [(r/10, 0.0) for r in range(11)]
        
        if relevant_points[0][0] > 0:
            max_precision = max([p[1] for p in relevant_points])
            relevant_points.insert(0, (0.0, max_precision))
        
        standard_recalls = [r/10 for r in range(11)]
        interpolated_curve = []
        
        for rj in standard_recalls:
            precisions_at_or_above = [p for r, p in relevant_points if r >= rj]
            if precisions_at_or_above:
                max_precision = max(precisions_at_or_above)
                interpolated_curve.append((rj, max_precision))
            else:
                interpolated_curve.append((rj, 0.0))
        
        return interpolated_curve
    
    def plot_precision_recall_curve(self, precision_recall_points, query_id, interpolated=False):
        """Trace la courbe Précision-Rappel pour une requête"""
        if interpolated:
            curve = self.interpolate_precision_recall(precision_recall_points)
            recalls = [p[0] for p in curve]
            precisions = [p[1] for p in curve]
            title = f"Courbe Précision-Rappel Interpolée - Requête {query_id}"
            filename = f"query_{query_id}_interpolated.png"
            save_dir = self.dirs['curves_interpolated']
        else:
            relevant_points = [p for p in precision_recall_points if p['is_relevant']]
            recalls = [p['recall'] for p in relevant_points]
            precisions = [p['precision'] for p in relevant_points]
            title = f"Courbe Précision-Rappel - Requête {query_id}"
            filename = f"query_{query_id}_non_interpolated.png"
            save_dir = self.dirs['curves_non_interpolated']
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Rappel', fontsize=12)
        plt.ylabel('Précision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_all_queries_curves(self, all_results, interpolated=False):
        """Trace toutes les courbes sur un même graphique"""
        plt.figure(figsize=(14, 8))
        
        for query_id, result in sorted(all_results.items()):
            if interpolated:
                curve = self.interpolate_precision_recall(result['precision_recall_curve'])
                recalls = [p[0] for p in curve]
                precisions = [p[1] for p in curve]
                label = f"Q{query_id}"
            else:
                relevant_points = [p for p in result['precision_recall_curve'] if p['is_relevant']]
                recalls = [p['recall'] for p in relevant_points]
                precisions = [p['precision'] for p in relevant_points]
                label = f"Q{query_id}"
            
            plt.plot(recalls, precisions, '-o', linewidth=1.5, markersize=4, alpha=0.7, label=label)
        
        plt.xlabel('Rappel', fontsize=12)
        plt.ylabel('Précision', fontsize=12)
        
        if interpolated:
            title = "Courbes Précision-Rappel Interpolées - Toutes les Requêtes"
            filename = "all_queries_interpolated.png"
        else:
            title = "Courbes Précision-Rappel - Toutes les Requêtes"
            filename = "all_queries_non_interpolated.png"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        
        save_path = os.path.join(self.dirs['all_queries'], filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def bm25_with_metrics(self, query_id, k1=1.2, b=0.75, verbose=False):
        """BM25 avec calcul de toutes les métriques"""
        query = None
        for q in self.queries:
            if q.query_id == query_id:
                query = q
                break
        
        if query is None:
            return None
        
        relevant_docs = set(self.relevance_judgments.get(query_id, []))
        total_relevant = len(relevant_docs)
        
        valid_terms = self.process_query(query.text)
        
        if not valid_terms:
            return None
        
        # Calculer RSV pour tous les documents
        rsv_scores = {}
        
        for doc_id in self.doc_ids:
            rsv = 0
            dl = self.doc_lengths.get(doc_id, 0)
            
            if dl == 0:
                rsv_scores[doc_id] = 0
                continue
            
            for term in valid_terms:
                tf = self.get_term_freq(term, doc_id)
                
                if tf > 0:
                    ni = self.compute_ni(term)
                    if ni == 0:
                        continue
                    
                    idf = math.log10((self.N - ni + 0.5) / (ni + 0.5))
                    normalization = 1 - b + b * (dl / self.avg_doc_length)
                    tf_component = (tf * (k1 + 1)) / (tf + k1 * normalization)
                    rsv += idf * tf_component
            
            rsv_scores[doc_id] = rsv
        
        # Trier par score décroissant
        ranking = sorted(rsv_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calcul progressif Précision/Rappel
        precision_recall_curve = []
        relevant_found = 0
        
        for rank, (doc_id, rsv) in enumerate(ranking, start=1):
            is_relevant = doc_id in relevant_docs
            
            if is_relevant:
                relevant_found += 1
                recall = relevant_found / total_relevant if total_relevant > 0 else 0
                precision = relevant_found / rank
                
                precision_recall_curve.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'rsv': rsv,
                    'is_relevant': True,
                    'recall': recall,
                    'precision': precision,
                    'relevant_found': relevant_found
                })
            else:
                precision_recall_curve.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'rsv': rsv,
                    'is_relevant': False,
                    'recall': None,
                    'precision': None,
                    'relevant_found': relevant_found
                })
        
        # Calculer toutes les métriques
        precision_final = relevant_found / len(ranking) if len(ranking) > 0 else 0
        recall_final = relevant_found / total_relevant if total_relevant > 0 else 0
        f_measure = self.calculate_f_measure(precision_final, recall_final)
        
        # P@K pour K = 5, 10, 15, 20, ..., jusqu'au total de documents
        k_values = list(range(5, len(ranking) + 1, 5))
        if len(ranking) not in k_values:
            k_values.append(len(ranking))
        p_at_k = self.calculate_precision_at_k(ranking, relevant_docs, k_values)
        
        # R-Precision
        r_precision = self.calculate_r_precision(ranking, relevant_docs)
        
        # Reciprocal Rank
        reciprocal_rank = self.calculate_reciprocal_rank(ranking, relevant_docs)
        
        # DCG@20 et nDCG@20
        dcg_at_20 = self.calculate_dcg_at_k(ranking, relevant_docs, rsv_scores, k=20)
        ndcg_at_20 = self.calculate_ndcg_at_k(ranking, relevant_docs, rsv_scores, k=20)
        
        # Average Precision
        avg_precision = self.calculate_average_precision(precision_recall_curve)
        interpolated_curve = self.interpolate_precision_recall(precision_recall_curve)
        avg_precision_interpolated = self.calculate_interpolated_average_precision(interpolated_curve)
        
        if verbose:
            print(f"\n{'='*120}")
            print(f"📊 REQUÊTE {query_id} - Évaluation BM25 Complète")
            print(f"{'='*120}")
            print(f"Texte: {query.text[:70]}...")
            print(f"Docs pertinents (R): {total_relevant}")
            print(f"{'='*120}\n")
            
            # Afficher tout le ranking avec les métriques
            print(f"{'Rang':<6} {'Doc':<8} {'RSV':<12} {'Pert.':<8} {'Rappel':<12} {'Précision':<12}")
            print(f"{'-'*120}")
            
            for point in precision_recall_curve:
                rank = point['rank']
                doc_id = point['doc_id']
                rsv = point['rsv']
                is_rel = point['is_relevant']
                recall = point['recall'] if is_rel else '-'
                precision = point['precision'] if is_rel else '-'
                
                if is_rel:
                    print(f"{rank:<6} {doc_id:<8} {rsv:<12.4f} {'✓':<8} {recall:<12.4f} {precision:<12.4f}")
                else:
                    print(f"{rank:<6} {doc_id:<8} {rsv:<12.4f} {'✗':<8} {recall:<12} {precision:<12}")
            
            print(f"\n{'='*120}")
            print(f"✅ MÉTRIQUES FINALES:")
            print(f"   - Docs pertinents trouvés: {relevant_found}/{total_relevant}")
            print(f"   - Rappel final: {recall_final:.4f}")
            print(f"   - Précision finale: {precision_final:.4f}")
            print(f"   - F-Mesure: {f_measure:.4f}")
            print(f"   - R-Precision: {r_precision:.4f}")
            print(f"   - Reciprocal Rank: {reciprocal_rank:.4f}")
            print(f"   - DCG@20: {dcg_at_20:.4f}")
            print(f"   - nDCG@20: {ndcg_at_20:.4f}")
            print(f"   - Average Precision (non interpolée): {avg_precision:.4f}")
            print(f"   - Average Precision (interpolée): {avg_precision_interpolated:.4f}")
            print(f"\n   📊 Précision@K:")
            for k in sorted(p_at_k.keys()):
                print(f"      P@{k}: {p_at_k[k]:.4f}")
            print(f"{'='*120}\n")
        
        return {
            'ranking': ranking,
            'relevant_docs': list(relevant_docs),
            'precision_recall_curve': precision_recall_curve,
            'interpolated_curve': interpolated_curve,
            'query_text': query.text,
            'total_relevant': total_relevant,
            'relevant_found': relevant_found,
            'precision_final': precision_final,
            'recall_final': recall_final,
            'f_measure': f_measure,
            'r_precision': r_precision,
            'reciprocal_rank': reciprocal_rank,
            'dcg_at_20': dcg_at_20,
            'ndcg_at_20': ndcg_at_20,
            'average_precision': avg_precision,
            'average_precision_interpolated': avg_precision_interpolated,
            'p_at_k': p_at_k,
            'rsv_scores': rsv_scores
        }
    
    def save_detailed_results(self, all_results, output_file="results/bm25_results.txt"):
        """Sauvegarde tous les résultats détaillés dans un fichier texte"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write("RÉSULTATS BM25 - ÉVALUATION COMPLÈTE\n")
            f.write("="*120 + "\n\n")
            
            for query_id in sorted(all_results.keys()):
                result = all_results[query_id]
                
                f.write("="*120 + "\n")
                f.write(f"REQUÊTE {query_id}\n")
                f.write("="*120 + "\n")
                f.write(f"Texte: {result['query_text']}\n")
                f.write(f"Documents pertinents (R): {result['total_relevant']}\n")
                f.write(f"Documents pertinents trouvés: {result['relevant_found']}\n\n")
                
                # Métriques globales
                f.write("MÉTRIQUES GLOBALES:\n")
                f.write("-"*120 + "\n")
                f.write(f"Rappel final:                     {result['recall_final']:.4f}\n")
                f.write(f"Précision finale:                 {result['precision_final']:.4f}\n")
                f.write(f"F-Mesure:                         {result['f_measure']:.4f}\n")
                f.write(f"R-Precision:                      {result['r_precision']:.4f}\n")
                f.write(f"Reciprocal Rank:                  {result['reciprocal_rank']:.4f}\n")
                f.write(f"DCG@20:                           {result['dcg_at_20']:.4f}\n")
                f.write(f"nDCG@20:                          {result['ndcg_at_20']:.4f}\n")
                f.write(f"Average Precision (non interp.):  {result['average_precision']:.4f}\n")
                f.write(f"Average Precision (interpolée):   {result['average_precision_interpolated']:.4f}\n\n")
                
                # P@K
                f.write("PRÉCISION@K:\n")
                f.write("-"*120 + "\n")
                for k in sorted(result['p_at_k'].keys()):
                    f.write(f"P@{k:<4} = {result['p_at_k'][k]:.4f}\n")
                f.write("\n")
                
                # Ranking complet
                f.write("CLASSEMENT COMPLET (1033 DOCUMENTS):\n")
                f.write("-"*120 + "\n")
                f.write(f"{'Rang':<6} {'Doc':<8} {'RSV':<15} {'Pert.':<8} {'Rappel':<12} {'Précision':<12}\n")
                f.write("-"*120 + "\n")
                
                for point in result['precision_recall_curve']:
                    rank = point['rank']
                    doc_id = point['doc_id']
                    rsv = point['rsv']
                    is_rel = point['is_relevant']
                    recall = f"{point['recall']:.4f}" if is_rel else "-"
                    precision = f"{point['precision']:.4f}" if is_rel else "-"
                    pert_mark = "✓" if is_rel else "✗"
                    
                    f.write(f"{rank:<6} {doc_id:<8} {rsv:<15.6f} {pert_mark:<8} {recall:<12} {precision:<12}\n")
                
                f.write("\n\n")
        
        print(f"\n✅ Résultats détaillés sauvegardés dans: {output_file}")
    
    def evaluate_all_queries(self, k1=1.2, b=0.75, verbose=False, plot_curves=True, save_results=True):
        """Évalue BM25 sur toutes les requêtes avec génération des courbes"""
        results = {}
        
        print("\n" + "="*120)
        print("🚀 ÉVALUATION BM25 - TOUTES LES REQUÊTES")
        print("="*120)
        print(f"Paramètres: k1={k1}, b={b}")
        print(f"Requêtes: {len(self.queries)} | Documents: {self.N}")
        print(f"Longueur moyenne: {self.avg_doc_length:.2f} termes")
        print("="*120)
        
        for query in self.queries:
            query_id = query.query_id
            result = self.bm25_with_metrics(query_id, k1, b, verbose=verbose)
            
            if result:
                results[query_id] = result
                
                if plot_curves:
                    self.plot_precision_recall_curve(result['precision_recall_curve'], query_id, interpolated=False)
                    self.plot_precision_recall_curve(result['precision_recall_curve'], query_id, interpolated=True)
                
                print(f"✅ Q{query_id:2d} | AP: {result['average_precision']:.4f} | "
                      f"R-Prec: {result['r_precision']:.4f} | "
                      f"RR: {result['reciprocal_rank']:.4f} | "
                      f"nDCG@20: {result['ndcg_at_20']:.4f}")
        
        if plot_curves:
            print("\n📊 Génération des courbes globales...")
            self.plot_all_queries_curves(results, interpolated=False)
            self.plot_all_queries_curves(results, interpolated=True)
            print(f"✅ Courbes sauvegardées dans: {self.dirs['base']}")
        
        # Calculer les moyennes globales
        map_score = sum(r['average_precision'] for r in results.values()) / len(results)
        mean_r_precision = sum(r['r_precision'] for r in results.values()) / len(results)
        mean_rr = sum(r['reciprocal_rank'] for r in results.values()) / len(results)
        mean_ndcg = sum(r['ndcg_at_20'] for r in results.values()) / len(results)
        
        print("\n" + "="*120)
        print("📈 MÉTRIQUES GLOBALES MOYENNES:")
        print(f"   - MAP (Mean Average Precision):  {map_score:.4f}")
        print(f"   - Mean R-Precision:              {mean_r_precision:.4f}")
        print(f"   - MRR (Mean Reciprocal Rank):    {mean_rr:.4f}")
        print(f"   - Mean nDCG@20:                  {mean_ndcg:.4f}")
        print("="*120)
        
        if save_results:
            self.save_detailed_results(results)
        
        return results


if __name__ == "__main__":
    import os
    from medline_parser import parse_med_all, parse_med_qry, parse_med_rel
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    MED_ALL_PATH = os.path.join(project_dir, "data", "MED.ALL")
    MED_QRY_PATH = os.path.join(project_dir, "data", "MED.QRY")
    MED_REL_PATH = os.path.join(project_dir, "data", "MED.REL")
    INVERTED_INDEX_PATH = os.path.join(project_dir, "data", "ouput", "inverted_index.txt")
    DOC_TERM_MATRIX_PATH = os.path.join(project_dir, "data", "ouput", "document_term_matrix.txt")
    
    print("📂 Chargement des données MEDLINE...")
    documents = parse_med_all(MED_ALL_PATH)
    queries = parse_med_qry(MED_QRY_PATH)
    relevance_judgments = parse_med_rel(MED_REL_PATH)
    
    print("\n🔧 Initialisation du modèle BM25...")
    bm25_model = BM25Model(
        inverted_index_path=INVERTED_INDEX_PATH,
        doc_term_matrix_path=DOC_TERM_MATRIX_PATH,
        documents=documents,
        queries=queries,
        relevance_judgments=relevance_judgments
    )
    
    # Exemple détaillé pour une requête
    print("\n" + "="*120)
    print("DÉMONSTRATION - Requête 1 détaillée")
    print("="*120)
    result_q1 = bm25_model.bm25_with_metrics(query_id=1, verbose=True)
    
    # Évaluation complète
    all_results = bm25_model.evaluate_all_queries(
        k1=1.2, 
        b=0.75, 
        verbose=False, 
        plot_curves=True,
        save_results=True
    )
    
    print("\n✅ Évaluation complète terminée!")
    print(f"📁 Fichiers générés:")
    print(f"   - results/bm25_results.txt (résultats détaillés)")
    print(f"   - results/figures/BM25/ (toutes les courbes)")