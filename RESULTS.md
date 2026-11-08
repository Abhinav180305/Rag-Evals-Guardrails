| Strategy | Runs | Correctness | Faithfulness | Relevancy | Ctx precision | Ctx recall | Guardrail | Refused (answerable) | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| dense | 100 | 0.72 | 0.88 | 0.78 | 0.89 | 0.84 | 0.87 | 11 | 13.4 |
| dense_rerank | 100 | 0.80 | 0.94 | 0.81 | 0.93 | 0.94 | 0.89 | 7 | 15.0 |
| hybrid_rerank | 100 | 0.84 | 0.96 | 0.80 | 0.96 | 0.94 | 0.93 | 6 | 15.1 |
| hybrid_rrf | 100 | 0.81 | 0.91 | 0.79 | 0.92 | 0.92 | 0.92 | 8 | 14.2 |
