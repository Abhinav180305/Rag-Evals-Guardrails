# RAG Evaluation Harness

A fully local framework for comparing retrieval strategies in a RAG pipeline. The same 100 questions go through each strategy, every answer is scored on six metrics, input/output guardrails are enforced, and each strategy is tracked as a separate experiment in LangSmith.

Everything except LangSmith tracking runs on one laptop CPU (Ollama for the models, FAISS for vectors, a local cross-encoder for reranking), so running it costs nothing.

## Results at a glance

100 questions per strategy, one run each. Generator: `llama3.2:3b`. Judge: `qwen2.5:7b`. Source: `make_results.py`.

| Strategy | Correctness | Faithfulness | Relevancy | Ctx precision | Ctx recall | Guardrail | Refused (of 80 answerable) | Median latency (s) |
|---|---|---|---|---|---|---|---|---|
| dense | 0.72 | 0.88 | 0.78 | 0.89 | 0.84 | 0.87 | 11 | 13.4 |
| hybrid_rrf | 0.81 | 0.91 | 0.79 | 0.92 | 0.92 | 0.92 | 8 | 14.2 |
| dense_rerank | 0.80 | 0.94 | 0.81 | 0.93 | 0.94 | 0.89 | 7 | 15.0 |
| hybrid_rerank | **0.84** | **0.96** | 0.80 | **0.96** | **0.94** | **0.93** | **6** | 15.1 |

The LangSmith web page may show values that differ in the second decimal place from this table, so quote one source consistently.

### Derived numbers (computed from the table above)

| Strategy | Correctness vs dense | Refusal rate on answerable questions | Latency vs dense |
|---|---|---|---|
| dense | baseline | 11 / 80 = 13.8% | baseline |
| hybrid_rrf | +0.09 | 8 / 80 = 10.0% | +0.8 s (+6%) |
| dense_rerank | +0.08 | 7 / 80 = 8.8% | +1.6 s (+12%) |
| hybrid_rerank | +0.12 | 6 / 80 = 7.5% | +1.7 s (+13%) |

Guardrail scores are means over 100 questions, so about 13 (dense), 8 (hybrid_rrf), 11 (dense_rerank) and 7 (hybrid_rerank) questions were handled wrongly. Subtracting the refusals of answerable questions leaves roughly 2, 0, 4 and 1 other misses, mostly unanswerable questions that still received an answer. These are approximate, because the scores are rounded, and I have not inspected the rows individually.

## How much to trust the differences

Correctness is averaged over 80 answerable questions. If per-question scores have a standard deviation of about 0.4 (an assumption; I did not measure it), the standard error of one mean is about 0.045, and an approximate 95% interval is ±0.09. Treat this as an approximation, not a test: it ignores that all strategies answer the same questions, which would make the comparison somewhat tighter.

- **Dense vs the other three:** gaps of 0.08 to 0.12 on correctness are suggestive and consistent across most metrics, but not conclusive.
- **Among the top three (0.80 to 0.84):** gaps of 0.04 or less are inside the noise. The ranking between them is not established by this data.
- Only one run per strategy was made, so run-to-run variation is unknown.

## Strategies

| Strategy | Retrieval |
|---|---|
| `dense` | Semantic search: embeddings (`nomic-embed-text`) in FAISS, top 3 |
| `hybrid_rrf` | BM25 keyword search (10 candidates) + dense search (10 candidates), merged with Reciprocal Rank Fusion, top 3 |
| `dense_rerank` | Dense search for 20 candidates, reordered by a cross-encoder (`ms-marco-MiniLM-L-6-v2`), top 3 |
| `hybrid_rerank` | BM25 (15) + dense (15) merged by RRF, best 20 reordered by the cross-encoder, top 3 |

Reciprocal Rank Fusion scores each chunk d as

    RRF(d) = Σ over rankings r of 1 / (60 + rank_r(d))

where rank_r(d) is the chunk's position (starting at 0, with 1 added in the code) in ranking r. A chunk ranked high by both BM25 and dense search wins.

Shared settings: chunk size 600 characters, overlap 80, 3 chunks passed to the generator, temperature 0. Only the retriever changes between strategies.

## Metrics and how each value is computed

The judge model labels items as true/false or gives a 0-10 rating. The scores themselves are computed in Python.

| Metric | Formula | Notes |
|---|---|---|
| Faithfulness | supported claims / total claims in the answer | Claims are marked by the judge as supported or not by the retrieved chunks |
| Context recall | ground-truth facts found in the chunks / total ground-truth facts | Low recall means retrieval missed information |
| Context precision | (1 / R) · Σ over relevant chunks k of (relevant chunks in top k) / k, where R = number of relevant chunks | Average precision; rewards ranking useful chunks first; 0 if no chunk is relevant |
| Answer relevancy | judge rating / 10 | How directly the answer addresses the question, ignoring correctness |
| Answer correctness | judge rating / 10 | Agreement with the ground truth; set to 0 when the model refuses an answerable question |
| Guardrail compliance | 1 if (refused == should_refuse), else 0 | Deterministic; no model involved |

Which questions each metric covers:
- **Guardrail compliance:** all 100 questions.
- **Answer correctness:** the 80 answerable questions (65 factual and 15 multi-hop), including refusals scored 0.
- **Faithfulness, relevancy, precision, recall:** answerable questions where the model gave a real answer. Refused questions are excluded, so these four averages are somewhat flattering, and a strategy that refuses more can look better on them. Read them next to the refusal count.
- **Unanswerable and adversarial questions** (20 total) are scored only by guardrail compliance.

Latency is the median seconds per question from LangSmith traces, including retrieval and generation.

## Guardrails

- **Input:** length limit (1,000 characters), regex-based prompt-injection blocking, PII redaction (email, phone, Aadhaar-style numbers, card numbers).
- **Prompt:** the model must answer only from the retrieved context and reply with a fixed refusal sentence when the answer is not there.
- **Output:** PII redaction and an empty-answer fallback.

These are pattern-based and can be bypassed by rewording. The adversarial questions use phrasings the patterns were written for, so a high guardrail score on them is not evidence of robustness.

## Dataset

100 questions generated from the source document by a local LLM (`qwen2.5:7b`): 65 factual, 15 multi-hop (built from neighbouring chunks), 10 unanswerable, 10 adversarial prompt-injection attempts.

A first version had about a third of its questions referring to "the passage" or "the text", which no retriever can answer. `clean_dataset.py` found those, regenerated them as self-contained questions, and uploaded a new dataset. All results above use the cleaned dataset.

## Findings

Plain semantic (dense) search was the weakest strategy: correctness 0.72, with 11 of 80 answerable questions refused. Adding BM25 (hybrid_rrf, 0.81), a cross-encoder reranker (dense_rerank, 0.80), or both (hybrid_rerank, 0.84) all improved on it, and refusals fell to 6 to 8. The three improved strategies are within 0.04 of each other, inside the noise expected from 80 scored questions, so this data does not rank them with confidence; hybrid_rerank was nominally best on correctness, faithfulness, context precision, context recall and guardrail compliance. Reranking added about 1.6 to 1.7 seconds of median latency (12 to 13%), and BM25 alone about 0.8 seconds (6%). Answer relevancy barely moved (0.78 to 0.81), which is expected because it measures whether the answer addresses the question, not whether retrieval found the right text.

## Failure analysis

- **Retrieval miss (dense):** for "What influenced behavior more according to Borgatta's study?", dense search returned a general chunk on attitudes and behaviour and the model refused; hybrid_rerank retrieved the chunk containing the answer.
- **Wrong page retrieved (dense):** for a question about a book edited by Ulrich and Mountjoy, dense search returned an author-index page listing the same names instead of the reference entry holding the title.
- **Generator refusal, cause unclear:** for a question about the drug imipramine, dense and hybrid_rerank ranked the same chunk first, yet only hybrid_rerank answered. This points to differences in the lower-ranked chunks or to the small generator's caution, not to ranking alone.
- **Questionable ground truth:** a question asking the year of a book "edited by W. McDougall" has a reference answer (1912) that I could not verify. Bibliographic questions like this are weak tests of understanding.

## Limitations

- One source document and 100 questions; small differences may be noise.
- Questions and ground-truth answers are written by an LLM from the same chunks the retriever searches, which tends to make them easier than real user questions, and some ground truths may be wrong.
- The judge is a local 7B model and is noisy. In a hand check of 10 judged rows I agreed with it on about 7.5 of 10, so treat scores as relative comparisons between strategies, not absolute accuracy.
- One run per strategy, so run-to-run variation is not measured.
- Guardrails are regex-based demonstrations, not a security layer.
- This is an evaluation harness, not a deployed application.

## Setup

1. Install [Ollama](https://ollama.com/download) and pull the models:
   ```
   ollama pull llama3.2:3b
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
2. `pip install -r requirements.txt`
3. Create `.env` from `.env.example` and add a free LangSmith key.
4. Put your own `.txt`, `.md` or text-based `.pdf` files in `data/docs/`.

## Run

```
python -m src.make_dataset         # generate questions, upload to LangSmith
python clean_dataset.py            # replace broken questions, upload the clean dataset
```
Then one strategy at a time (PowerShell shown):
```
$env:DEV_N=0                       # 0 = all questions; default is a 10-question smoke test
$env:STRATEGY="dense"              # dense | hybrid_rrf | dense_rerank | hybrid_rerank
python -m src.run_eval
```
Build the results table with `python make_results.py`. On a laptop CPU, expect about 1 to 1.5 hours per strategy for 100 questions with a 7B judge.

## Layout

```
src/
  config.py        models and settings (override via .env)
  strategies.py    the four retrieval strategies
  guardrails.py    input/output checks
  pipeline.py      guarded RAG pipeline, traced in LangSmith
  evaluators.py    the six metrics
  make_dataset.py  question generation
  run_eval.py      runs a strategy over the dataset
clean_dataset.py   rewrites broken questions
make_results.py    builds the results table
```

The source document and the question files are not included in this repository.
