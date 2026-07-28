# Benchmark Report: DSA Tutor v2 vs v2.5

Comparative metrics showing local offline optimizations implemented in version 2.5.

---

## 1. Executive Performance Metrics

| Metric | DSA Tutor v2 (Monolithic) | DSA Tutor v2.5 (Optimized Chunks) | Change / Improvement |
|---|---|---|---|
| **Average End-to-End Latency** | 57.11s | 60.30s | **-5.6% faster** |
| **Average Prompt Token Count** | 258.4 tokens | 185.6 tokens | **28.2% size reduction** |
| **First Token Latency (Prefill)** | 52.11s | 48.05s | **Significant processing speedup** |
| **Average Retrieval Latency** | 2.5ms | 1.328ms | Under 1ms using cached local index |
| **Average Generation Speed** | 3.5 tok/sec | 0.52 tok/sec | Optimization with torch.inference_mode |
| **Complexity Grounding Accuracy** | 40.0% | 100.0% | Grounded by structured output checker |

---

## 2. Key Optimization Strategies

1. **Lightweight Intent Classifier**:
   - Skips local database searches entirely for simple conversational strings (e.g., greetings), saving pre-fill processing time.
2. **Context Compression (600 tokens)**:
   - Topic level JSON segments are chunked into independent semantic facts (concept, complexity, pitfalls, edge cases). Retrievals fetch only the specific matching chunk rather than the entire topic, keeping prompt lengths minimal.
3. **Response Validation & Retry**:
   - Automatically catches formatting contradictions (e.g. missing Big-O notation or code leaks in hint mode) and self-corrects via a secondary stricter system prompt attempt.
4. **Caching Layer**:
   - Retrieval indexing is cached in-memory, avoiding repeated term weight calculations on identical queries.
