import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
from collections import defaultdict, Counter

# ============================================================================
# PREPROCESSING (FROM LAB 1)
# ============================================================================
print("=" * 80)
print("LAB 4 - LANGUAGE MODELS FOR INFORMATION RETRIEVAL")
print("=" * 80)

# Read documents
documents = {}
collection_path = r"C:\Users\pc\Desktop\RI\Collection"
for filename in os.listdir(collection_path):
    if filename.endswith(".txt"):
        doc_id = filename.split(".")[0]
        with open(os.path.join(collection_path, filename), "r", encoding="utf-8") as f:
            documents[doc_id] = f.read()

print(f"Loaded {len(documents)} documents\n")

# Tokenization
ExpReg = RegexpTokenizer(
    r'(?:[A-Za-z]\.)+|'
    r'[A-Za-z]+[\-@]\d+(?:\.\d+)?|'
    r'\d+(?:[\.\,\-]\d+)*%?|'
    r'[A-Za-z]+'
)

# Remove stop words and apply Porter stemmer
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

terms_processed = {}
for doc_id, content in documents.items():
    tokens = [term.lower() for term in ExpReg.tokenize(content)]
    filtered = [term for term in tokens if term not in stop_words]
    stemmed = [porter.stem(term) for term in filtered]
    terms_processed[doc_id] = stemmed

print("Preprocessing completed!\n")

# ============================================================================
# STEP 1: BUILD VOCABULARIES
# ============================================================================
print("=" * 80)
print("STEP 1: BUILD VOCABULARIES")
print("=" * 80)

# Vocabulary for each document
doc_vocab = {}
for doc_id, terms in terms_processed.items():
    doc_vocab[doc_id] = set(terms)
    print(f"Document {doc_id}: |V| = {len(doc_vocab[doc_id])}")

# Collection vocabulary
collection_vocab = set()
for terms in terms_processed.values():
    collection_vocab.update(terms)
print(f"\nCollection vocabulary: |V| = {len(collection_vocab)}")

# ============================================================================
# STEP 2: COMPUTE TERM FREQUENCIES tf(w,d)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: COMPUTE TERM FREQUENCIES tf(w,d)")
print("=" * 80)

doc_tf = {}
doc_lengths = {}
for doc_id, terms in terms_processed.items():
    doc_tf[doc_id] = Counter(terms)
    doc_lengths[doc_id] = len(terms)
    print(f"Document {doc_id}: {doc_lengths[doc_id]} terms")

# ============================================================================
# STEP 3: COMPUTE COLLECTION FREQUENCIES cf(w,C)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: COMPUTE COLLECTION FREQUENCIES cf(w,C)")
print("=" * 80)

collection_tf = Counter()
total_terms_in_collection = 0
for terms in terms_processed.values():
    collection_tf.update(terms)
    total_terms_in_collection += len(terms)

print(f"Total terms in collection: {total_terms_in_collection}")
print(f"Unique terms: {len(collection_tf)}")

# ============================================================================
# QUERY PREPROCESSING
# ============================================================================
def preprocess_query(query_text):
    """Preprocess query using same pipeline as documents"""
    tokens = [term.lower() for term in ExpReg.tokenize(query_text)]
    filtered = [term for term in tokens if term not in stop_words]
    stemmed = [porter.stem(term) for term in filtered]
    return stemmed

queries = {
    "q1": "large language models for information retrieval and ranking",
    "q2": "LLM for information retrieval and Ranking",
    "q3": "query Reformulation in information retrieval",
    "q4": "ranking Documents",
    "q5": "Optimizing recommendation systems with LLMs by leveraging item metadata"
}

processed_queries = {}
print("\n" + "=" * 80)
print("QUERY PREPROCESSING")
print("=" * 80)
for q_id, q_text in queries.items():
    processed_queries[q_id] = preprocess_query(q_text)
    print(f"{q_id}: {processed_queries[q_id]}")

# ============================================================================
# MODEL 1: UNSMOOTHED LANGUAGE MODEL (MLE)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: UNSMOOTHED LANGUAGE MODEL (MLE)")
print("=" * 80)

def compute_mle_score(query_terms, doc_id):
    """P(Q|D) = ∏ P(w|D) where P(w|D) = tf(w,d) / |D|"""
    score = 0.0
    for term in query_terms:
        tf = doc_tf[doc_id].get(term, 0)
        if tf == 0:
            return float('-inf')  # Zero probability
        prob = tf / doc_lengths[doc_id]
        score += math.log(prob)
    return score

mle_results = {}
for q_id, q_terms in processed_queries.items():
    scores = {}
    for doc_id in documents.keys():
        scores[doc_id] = compute_mle_score(q_terms, doc_id)
    mle_results[q_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{q_id}: {queries[q_id]}")
    for doc_id, score in mle_results[q_id]:
        print(f"  {doc_id}: {score:.6f}")

# ============================================================================
# MODEL 2: ADD-1 (LAPLACE) SMOOTHING
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: ADD-1 (LAPLACE) SMOOTHING")
print("=" * 80)

def compute_laplace_score(query_terms, doc_id, vocab_size):
    """P(w|D) = (tf(w,d) + 1) / (|D| + |V|)"""
    score = 0.0
    for term in query_terms:
        tf = doc_tf[doc_id].get(term, 0)
        prob = (tf + 1) / (doc_lengths[doc_id] + vocab_size)
        score += math.log(prob)
    return score

V = len(collection_vocab)
laplace_results = {}
for q_id, q_terms in processed_queries.items():
    scores = {}
    for doc_id in documents.keys():
        scores[doc_id] = compute_laplace_score(q_terms, doc_id, V)
    laplace_results[q_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{q_id}: {queries[q_id]}")
    for doc_id, score in laplace_results[q_id]:
        print(f"  {doc_id}: {score:.6f}")

# ============================================================================
# MODEL 3: GOOD-TURING SMOOTHING (APPROXIMATED)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: GOOD-TURING SMOOTHING (APPROXIMATED)")
print("=" * 80)

def compute_good_turing_score(query_terms, doc_id):
    """Good-Turing with approximation for unseen terms"""
    # Count frequency of frequencies
    freq_counts = Counter(doc_tf[doc_id].values())
    N_c = {c: freq_counts[c] for c in freq_counts}
    
    # N1 = number of terms appearing exactly once
    N1 = N_c.get(1, 0)
    N0 = len(collection_vocab) - len(doc_tf[doc_id])  # Unseen terms
    
    score = 0.0
    for term in query_terms:
        c = doc_tf[doc_id].get(term, 0)
        
        if c == 0:
            # Unseen term
            if N0 > 0:
                prob = N1 / (doc_lengths[doc_id] * N0)
            else:
                prob = 1e-10
        else:
            # Seen term: c* = (c+1) * N_{c+1} / N_c
            N_c_plus_1 = N_c.get(c + 1, 0)
            if N_c_plus_1 > 0:
                c_star = (c + 1) * N_c_plus_1 / N_c[c]
            else:
                c_star = c
            prob = c_star / doc_lengths[doc_id]
        
        score += math.log(max(prob, 1e-10))
    return score

gt_results = {}
for q_id, q_terms in processed_queries.items():
    scores = {}
    for doc_id in documents.keys():
        scores[doc_id] = compute_good_turing_score(q_terms, doc_id)
    gt_results[q_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{q_id}: {queries[q_id]}")
    for doc_id, score in gt_results[q_id]:
        print(f"  {doc_id}: {score:.6f}")

# ============================================================================
# MODEL 4: JELINEK-MERCER SMOOTHING (λ = 0.4)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 4: JELINEK-MERCER SMOOTHING (λ = 0.4)")
print("=" * 80)

def compute_jm_score(query_terms, doc_id, lambda_param=0.4):
    """P(w|D) = λ * P_ml(w|D) + (1-λ) * P_ml(w|C)"""
    score = 0.0
    for term in query_terms:
        # Document probability
        p_doc = doc_tf[doc_id].get(term, 0) / doc_lengths[doc_id]
        # Collection probability
        p_coll = collection_tf.get(term, 0) / total_terms_in_collection
        # Interpolated probability
        prob = lambda_param * p_doc + (1 - lambda_param) * p_coll
        score += math.log(max(prob, 1e-10))
    return score

jm_results = {}
for q_id, q_terms in processed_queries.items():
    scores = {}
    for doc_id in documents.keys():
        scores[doc_id] = compute_jm_score(q_terms, doc_id, 0.4)
    jm_results[q_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{q_id}: {queries[q_id]}")
    for doc_id, score in jm_results[q_id]:
        print(f"  {doc_id}: {score:.6f}")

# ============================================================================
# MODEL 5: DIRICHLET SMOOTHING
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 5: DIRICHLET SMOOTHING")
print("=" * 80)

# Calculate average document length
N_avg = sum(doc_lengths.values()) / len(doc_lengths)
mu = 0.3 * N_avg
print(f"Average document length: {N_avg:.2f}")
print(f"Dirichlet parameter μ: {mu:.2f}")

def compute_dirichlet_score(query_terms, doc_id, mu):
    """P(w|D) = (tf(w,d) + μ * P(w|C)) / (|D| + μ)"""
    score = 0.0
    for term in query_terms:
        tf = doc_tf[doc_id].get(term, 0)
        p_coll = collection_tf.get(term, 0) / total_terms_in_collection
        prob = (tf + mu * p_coll) / (doc_lengths[doc_id] + mu)
        score += math.log(max(prob, 1e-10))
    return score

dirichlet_results = {}
for q_id, q_terms in processed_queries.items():
    scores = {}
    for doc_id in documents.keys():
        scores[doc_id] = compute_dirichlet_score(q_terms, doc_id, mu)
    dirichlet_results[q_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{q_id}: {queries[q_id]}")
    for doc_id, score in dirichlet_results[q_id]:
        print(f"  {doc_id}: {score:.6f}")

# ============================================================================
# SUMMARY: ALL MODELS COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: RANKING COMPARISON FOR ALL MODELS")
print("=" * 80)

models = {
    "MLE (Unsmoothed)": mle_results,
    "Laplace (Add-1)": laplace_results,
    "Good-Turing": gt_results,
    "Jelinek-Mercer (λ=0.4)": jm_results,
    "Dirichlet (μ={:.2f})".format(mu): dirichlet_results
}

for q_id in processed_queries.keys():
    print(f"\n{'='*80}")
    print(f"Query {q_id}: {queries[q_id]}")
    print(f"{'='*80}")
    
    for model_name, results in models.items():
        ranking = [doc_id for doc_id, score in results[q_id]]
        print(f"{model_name:30s}: {' > '.join(ranking)}")

print("\n" + "=" * 80)
print("LAB 4 COMPLETED!")
print("=" * 80)