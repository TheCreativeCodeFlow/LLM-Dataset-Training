#!/usr/bin/env python3
"""
scripts/vector_store.py

Lightweight offline TF-IDF and Cosine Similarity vector retriever.
Runs fully locally in pure Python/NumPy, bypassing heavy DLL files blocked by security policies.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple

KB_PATH = "data/knowledge_base/dsa_kb.json"
INDEX_PATH = "data/knowledge_base/index.json"

class VectorStore:
    def __init__(self, index_path: str = INDEX_PATH, kb_path: str = KB_PATH):
        self.index_path = index_path
        self.kb_path = kb_path
        self.documents: List[Dict[str, Any]] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[np.ndarray] = []
        
        # Load knowledge base and build local index
        self.load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        """Cleans and tokenizes text into lowercased alphanumeric words."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def load_and_index(self):
        """Loads raw DSA JSON documents and indexes them in a TF-IDF vector matrix."""
        if not os.path.exists(self.kb_path):
            # Create index parent folders if missing
            os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
            print(f"[VectorStore] WARNING: KB file missing at {self.kb_path}")
            return
            
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
            
        print(f"[VectorStore] Indexing {len(self.documents)} DSA topics locally using pure TF-IDF...")
        
        # 1. Extract tokens
        all_tokens = []
        doc_tokens = []
        for doc in self.documents:
            doc_text = (
                f"{doc['topic']} {doc['concept']} {doc['intuition']} "
                f"{' '.join(doc['common_mistakes'])} {' '.join(doc['edge_cases'])} "
                f"{doc['interview_tips']}"
            )
            tokens = self._tokenize(doc_text)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)
            
        # 2. Build vocabulary
        unique_tokens = sorted(list(set(all_tokens)))
        self.vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
        
        # 3. Compute IDF
        num_docs = len(self.documents)
        for tok in self.vocab:
            df = sum(1 for tokens in doc_tokens if tok in tokens)
            self.idf[tok] = np.log((num_docs + 1) / (df + 1)) + 1.0
            
        # 4. Generate Document Vectors
        self.doc_vectors = []
        for tokens in doc_tokens:
            vec = np.zeros(len(self.vocab), dtype=np.float32)
            for tok in tokens:
                if tok in self.vocab:
                    vec[self.vocab[tok]] += 1
            # Apply IDF weights
            for tok, idx in self.vocab.items():
                vec[idx] *= self.idf[tok]
            # L2 Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.doc_vectors.append(vec)
            
        # Write clean document list copy to index.json for serialization compatibility
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump({"documents": self.documents}, f, indent=2)
        print(f"[VectorStore] Successfully indexed and saved configuration index to {self.index_path}")

    def retrieve(self, query: str, top_k: int = 2, topic_filter: str = None) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieves top-k relevant DSA documents based on Cosine Similarity of TF-IDF vectors."""
        if len(self.documents) == 0:
            return []
            
        query_tokens = self._tokenize(query)
        query_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in query_tokens:
            if tok in self.vocab:
                query_vec[self.vocab[tok]] += 1
                
        # Apply IDF weights
        for tok, idx in self.vocab.items():
            query_vec[idx] *= self.idf[tok]
            
        # L2 Normalize
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
            
        # Calculate cosine similarities
        scores = []
        for idx, doc in enumerate(self.documents):
            if topic_filter and doc["topic"].lower() != topic_filter.lower():
                continue
            doc_vec = self.doc_vectors[idx]
            sim = float(np.dot(query_vec, doc_vec))
            scores.append((doc, sim))
            
        # Sort by similarity score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

if __name__ == "__main__":
    print("=== Testing Local TF-IDF Vector Store ===")
    store = VectorStore()
    
    # Test query
    test_query = "Explain binary search tree height balancing AVL rules."
    print(f"\nQuery: '{test_query}'")
    results = store.retrieve(test_query, top_k=1)
    if len(results) > 0:
        doc, score = results[0]
        print(f"Match: {doc['topic']} (Score: {score:.3f})")
        print(f"Concept: {doc['concept']}")
    else:
        print("No match found.")
