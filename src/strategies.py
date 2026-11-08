"""3 retrieval strategies. Each takes (query) -> list[Document] of length TOP_K."""
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import config as C

_cache = {}

def load_chunks():
    if "chunks" in _cache:
        return _cache["chunks"]
    docs = []
    for p in sorted(Path(C.DOCS_DIR).rglob("*")):
        if p.suffix in {".txt", ".md"}:
            docs += TextLoader(str(p), encoding="utf-8").load()
        elif p.suffix == ".pdf":
            docs += PyPDFLoader(str(p)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    _cache["chunks"] = chunks
    return chunks

def vectorstore():
    if "vs" in _cache:
        return _cache["vs"]
    emb = C.embeddings()
    if Path(C.INDEX_DIR).exists():
        vs = FAISS.load_local(C.INDEX_DIR, emb, allow_dangerous_deserialization=True)
    else:
        chunks = load_chunks()
        vs, B = None, 8
        for i in range(0, len(chunks), B):
            batch = chunks[i:i + B]
            if vs is None:
                vs = FAISS.from_documents(batch, emb)
            else:
                vs.add_documents(batch)
            print(f"  embedded {min(i + B, len(chunks))}/{len(chunks)} chunks", flush=True)
        vs.save_local(C.INDEX_DIR)
    _cache["vs"] = vs
    return vs

# ---- Strategy 1: naive dense ----
def dense(query):
    return vectorstore().similarity_search(query, k=C.TOP_K)

# ---- Strategy 2: hybrid BM25 + dense with Reciprocal Rank Fusion ----
def hybrid(query, pool=10, rrf_k=60):
    if "bm25" not in _cache:
        _cache["bm25"] = BM25Retriever.from_documents(load_chunks(), k=pool)
    lists = [_cache["bm25"].invoke(query), vectorstore().similarity_search(query, k=pool)]
    scores, by_id = {}, {}
    for lst in lists:
        for rank, d in enumerate(lst):
            cid = d.metadata["chunk_id"]
            by_id[cid] = d
            scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
    top = sorted(scores, key=scores.get, reverse=True)[: C.TOP_K]
    return [by_id[i] for i in top]

# ---- Strategy 3: dense (wide) + cross-encoder rerank ----
def rerank(query, pool=20):
    if "ce" not in _cache:
        from sentence_transformers import CrossEncoder
        _cache["ce"] = CrossEncoder(C.RERANK_MODEL)
    cands = vectorstore().similarity_search(query, k=pool)
    s = _cache["ce"].predict([(query, d.page_content) for d in cands])
    ranked = [d for _, d in sorted(zip(s, cands), key=lambda x: x[0], reverse=True)]
    return ranked[: C.TOP_K]

# ---- Strategy 4: hybrid (BM25 + dense, RRF) -> cross-encoder rerank ----
def hybrid_rerank(query, pool=15, rrf_k=60):
    if "bm25_wide" not in _cache:
        _cache["bm25_wide"] = BM25Retriever.from_documents(load_chunks(), k=pool)
    if "ce" not in _cache:
        from sentence_transformers import CrossEncoder
        _cache["ce"] = CrossEncoder(C.RERANK_MODEL)
    lists = [_cache["bm25_wide"].invoke(query), vectorstore().similarity_search(query, k=pool)]
    scores, by_id = {}, {}
    for lst in lists:
        for rank, d in enumerate(lst):
            cid = d.metadata["chunk_id"]
            by_id[cid] = d
            scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
    cands = [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)[:20]]
    s = _cache["ce"].predict([(query, d.page_content) for d in cands])
    ranked = [d for _, d in sorted(zip(s, cands), key=lambda x: x[0], reverse=True)]
    return ranked[: C.TOP_K]

STRATEGIES = {"dense": dense, "hybrid_rrf": hybrid, "dense_rerank": rerank, "hybrid_rerank": hybrid_rerank}
