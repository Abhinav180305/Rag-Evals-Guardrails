import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---- Models (local via Ollama). Override any of these from .env without editing code ----
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2:3b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:3b")          # DEV default (fast). Final run: JUDGE_MODEL=qwen2.5:7b
DATASET_MODEL = os.getenv("DATASET_MODEL", "qwen2.5:7b")      # one-time question generation
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Optional: send ONLY the judging to Gemini free tier (needs GOOGLE_API_KEY). Chunks leave your machine. ----
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "ollama")        # "ollama" | "gemini"
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash")

# ---- CPU-friendly retrieval settings ----
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K = 3                          # every strategy returns exactly K chunks

DATASET_NAME = os.getenv("DATASET_NAME", "rag-eval-100")
DOCS_DIR = "data/docs"
INDEX_DIR = "index"
REFUSAL = "I don't know based on the provided documents."
DEV_N = int(os.getenv("DEV_N", 10))   # 10 = quick dev run. Set DEV_N=0 for the full 100 questions.

def chat(model, json_mode=False, temperature=0.0):
    return ChatOllama(model=model, temperature=temperature, num_ctx=4096,
                      format="json" if json_mode else None)

def judge_llm():
    if JUDGE_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.rate_limiters import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(requests_per_second=0.2)   # ~12 req/min, under the free 15 RPM
        return ChatGoogleGenerativeAI(model=GEMINI_JUDGE_MODEL, temperature=0, max_retries=6,
                                      response_mime_type="application/json", rate_limiter=limiter)
    return chat(JUDGE_MODEL, json_mode=True)

def embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)
