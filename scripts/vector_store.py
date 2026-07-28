#!/usr/bin/env python3
"""
scripts/vector_store.py

Optimized Hybrid Offline Retriever for DSA Tutor v2.5.
Combines Cosine Similarity over TF-IDF vectors with Keyword Overlap matching.
Features LRU caching, metadata filtering, chunk-level retrieval, and context compression.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple

CHUNKS_PATH = "data/knowledge_base/dsa_kb_chunks.json"
INDEX_PATH = "data/knowledge_base/index.json"

class VectorStore:
    _instance = None  # Singleton instance to implement index cache

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, chunks_path: str = CHUNKS_PATH):
        if self._initialized:
            return
            
        self.chunks_path = chunks_path
        self.chunks: List[Dict[str, Any]] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.chunk_vectors: List[np.ndarray] = []
        
        # In-memory query-retrieval cache (Phase 8)
        self.retrieval_cache: Dict[str, Any] = {}
        
        # Load chunks and build index
        self.load_and_index()
        self._initialized = True

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-z0-9]+', text.lower())

    def load_and_index(self):
        """Loads semantic chunks and builds TF-IDF index cache in memory."""
        if not os.path.exists(self.chunks_path):
            print(f"[VectorStore] WARNING: Chunk file missing at {self.chunks_path}")
            return
            
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        print(f"[VectorStore] Indexing {len(self.chunks)} semantic chunks locally using Hybrid TF-IDF...")
        
        # 1. Tokenize all chunks
        all_tokens = []
        chunk_tokens = []
        for chunk in self.chunks:
            chunk_text = f"{chunk['topic']} {chunk['type']} {chunk['content']} {' '.join(chunk['keywords'])}"
            tokens = self._tokenize(chunk_text)
            chunk_tokens.append(tokens)
            all_tokens.extend(tokens)
            
        # 2. Build vocabulary
        unique_tokens = sorted(list(set(all_tokens)))
        self.vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
        
        # 3. Compute IDF
        num_docs = len(self.chunks)
        for tok in self.vocab:
            df = sum(1 for tokens in chunk_tokens if tok in tokens)
            self.idf[tok] = np.log((num_docs + 1) / (df + 1.0)) + 1.0
            
        # 4. Generate Chunks Vector Matrix
        self.chunk_vectors = []
        for tokens in chunk_tokens:
            vec = np.zeros(len(self.vocab), dtype=np.float32)
            for tok in tokens:
                if tok in self.vocab:
                    vec[self.vocab[tok]] += 1
            for tok, idx in self.vocab.items():
                vec[idx] *= self.idf[tok]
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.chunk_vectors.append(vec)
            
        print("[VectorStore] Index creation complete.")

    def clear_cache(self):
        """Manually invalidates the retrieval cache."""
        self.retrieval_cache.clear()
        print("[VectorStore] Retrieval cache cleared successfully.")

    def retrieve(self, query: str, top_k: int = 2, topic_filter: str = None, mode_filter: str = None, max_tokens: int = 600) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid retrieval combining TF-IDF similarity (60%) and Keyword Overlap matching (40%).
        Applies metadata filtering and context compression down to max_tokens budget.
        """
        # Cache Lookup
        cache_key = f"{query.strip().lower()}_k{top_k}_t{topic_filter}_m{mode_filter}_t{max_tokens}"
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
            
        if len(self.chunks) == 0:
            return []
            
        query_tokens = self._tokenize(query)
        query_set = set(query_tokens)
        
        # 1. Cosine Similarity Vector Calculation (TF-IDF)
        query_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in query_tokens:
            if tok in self.vocab:
                query_vec[self.vocab[tok]] += 1
        for tok, idx in self.vocab.items():
            query_vec[idx] *= self.idf[tok]
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
            
        scores = []
        for idx, chunk in enumerate(self.chunks):
            # Apply topic filter if specified
            if topic_filter and chunk["topic"].lower() != topic_filter.lower():
                continue
                
            # Cosine semantic score
            semantic_score = float(np.dot(query_vec, self.chunk_vectors[idx]))
            
            # Keyword match overlap score (normalized by length of query keywords)
            chunk_keywords = set(chunk["keywords"])
            overlap = len(query_set.intersection(chunk_keywords))
            keyword_score = overlap / len(query_set) if len(query_set) > 0 else 0.0
            
            # Metadata filter boost: Boost chunks matching user tutor mode
            mode_boost = 0.0
            if mode_filter:
                mode_mapping = {
                    "complexity_analyst": "complexity",
                    "debugging_mentor": "mistake",
                    "code_reviewer": "mistake",
                    "hint_generator": "concept"
                }
                target_type = mode_mapping.get(mode_filter)
                if target_type and chunk["type"] == target_type:
                    mode_boost = 0.15
                    
            # Weighted hybrid score calculation
            hybrid_score = (0.6 * semantic_score) + (0.4 * keyword_score) + mode_boost
            scores.append((chunk, hybrid_score))
            
        # Sort and select top candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        raw_candidates = scores[:top_k]
        
        # 2. Context Compression & Deduplication (600 token limit)
        compressed_candidates = []
        total_word_count = 0
        word_limit = int(max_tokens * 0.75) # Approximation: ~450 words for 600 tokens
        
        seen_contents = set()
        
        for chunk, score in raw_candidates:
            clean_content = chunk["content"].strip()
            # Deduplicate matching strings
            if clean_content in seen_contents:
                continue
                
            words = clean_content.split()
            if total_word_count + len(words) > word_limit:
                # Truncate content to fit exact budget
                allowed_words = word_limit - total_word_count
                if allowed_words <= 0:
                    break
                truncated_content = " ".join(words[:allowed_words]) + " [truncated]"
                chunk_copy = chunk.copy()
                chunk_copy["content"] = truncated_content
                compressed_candidates.append((chunk_copy, score))
                break
            else:
                compressed_candidates.append((chunk, score))
                total_word_count += len(words)
                seen_contents.add(clean_content)
                
        # Write to LRU cache (simple key-value storage)
        if len(self.retrieval_cache) > 200:
            # Clear oldest keys if cache grows large
            self.retrieval_cache.pop(next(iter(self.retrieval_cache)))
        self.retrieval_cache[cache_key] = compressed_candidates
        
        return compressed_candidates

if __name__ == "__main__":
    print("=== Testing Hybrid Vector Store ===")
    store = VectorStore()
    
    # Test query
    q = "Time complexity of AVL tree balancing rotations"
    print(f"\nQuery: '{q}'")
    results = store.retrieve(q, top_k=2, mode_filter="complexity_analyst")
    for chunk, score in results:
        print(f"\n[Matched Chunk] ID: {chunk['chunk_id']} (Hybrid Score: {score:.3f})")
        print(f"Content: {chunk['content']}")
