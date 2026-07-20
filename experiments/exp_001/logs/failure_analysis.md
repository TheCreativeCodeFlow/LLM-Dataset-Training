# Failure Analysis Report

- **Total Benchmark Samples:** 100
- **Successful Responses:** 0
- **Failed Responses:** 100

## Top Failure Categories

| Failure Category | Count | Percentage |
| --- | --- | --- |
| incorrect concept explanation | 100 | 100.0% |
| poor hint progression | 20 | 20.0% |
| incorrect complexity explanation | 20 | 20.0% |
| incomplete response | 10 | 10.0% |
| solution leakage | 4 | 4.0% |

## Failures by Topic

| Topic | Failures Count |
| --- | --- |
| Arrays | 10 |
| Strings | 10 |
| Hash Maps | 10 |
| Linked Lists | 10 |
| Trees | 10 |
| Graphs | 10 |
| Binary Search | 10 |
| Sliding Window | 10 |
| Dynamic Programming | 10 |
| Greedy | 10 |

## Failures by Conversation Type

| Conversation Type | Failures Count |
| --- | --- |
| concept explanation | 20 |
| hint generation | 20 |
| bug diagnosis | 20 |
| complexity analysis | 20 |
| interview follow-up | 20 |

## Base vs. Fine-Tuned Comparison

| Metric | Base Model | Fine-Tuned Model | Improvement |
| --- | --- | --- | --- |
| **Avg Latency** | 0.79 s | 1.02 s | -29.1% |
| **Educational Score** | 2.36 | 2.39 | +1.6% |
| **Failure Rate (per item)** | 1.54 | 1.45 | +5.8% |
| **Complexity Accuracy** | 2.00 | 2.00 | +0.0% |
| **Avg Response Length** | 256.0 tokens | 256.0 tokens | +0.0% |
| **Hint Solution Leakage Count** | 4 | 1 | +3 |
