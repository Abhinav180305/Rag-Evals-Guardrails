"""Replace questions that mention 'passage/excerpt/...' with self-contained ones; upload as a NEW LangSmith dataset."""
import json, random, re, collections, warnings
warnings.filterwarnings("ignore")
from langsmith import Client
from src import config as C
from src.strategies import load_chunks

random.seed(11)
NEW_DATASET = "rag-eval-clean"
BAD = re.compile(r"\b(passages?|excerpts?|the text|the document|the following|the above|the given|the provided|the context)\b", re.I)
RULES = ("The question must be fully self-contained: a reader who has never seen the source must understand it. "
         "Name the specific person, theory, study, drug or term it is about. "
         "Never use words like passage, text, excerpt, document or section, and never write 'the author' or 'the study' without naming it. "
         "Do not mention that a source exists.")
llm = C.chat(C.DATASET_MODEL, json_mode=True, temperature=0.4)

def gen(prompt):
    for _ in range(2):
        try:
            r = json.loads(llm.invoke(prompt + "\nRespond with JSON only.").content)
            if isinstance(r, dict) and r:
                return r
        except Exception:
            pass
    return None

def good_chunk(c):
    t = c.page_content
    return len(t) > 250 and sum(ch.isalpha() for ch in t) / len(t) > 0.75

def ok_q(q):
    q = str(q).strip()
    return len(q) >= 25 and q.endswith("?") and not BAD.search(q)

def collect(n, make, label):
    rows, tries = [], 0
    while len(rows) < n and tries < n * 8:
        tries += 1
        row = make()
        if row:
            rows.append(row)
            print(f"  {label}: {len(rows)}/{n}", flush=True)
    return rows

chunks = load_chunks()
good = [i for i, c in enumerate(chunks) if good_chunk(c)]

def make_factual():
    c = chunks[random.choice(good)]
    r = gen(f"Write ONE question answerable from the source below, and a short answer using only facts stated in it. {RULES}\n"
            f'Return {{"question": ..., "answer": ...}}\nSOURCE:\n{c.page_content}')
    if r and ok_q(r.get("question", "")) and r.get("answer"):
        return {"question": str(r["question"]).strip(), "ground_truth": str(r["answer"]).strip(), "type": "factual"}

def make_multi():
    i = random.choice(good)
    if i + 1 >= len(chunks) or (i + 1) not in set(good) or chunks[i].metadata.get("source") != chunks[i + 1].metadata.get("source"):
        return None
    a, b = chunks[i], chunks[i + 1]
    r = gen(f"Write ONE question that needs facts from BOTH sources below, and a short answer using only facts stated in them. {RULES}\n"
            f'Return {{"question": ..., "answer": ...}}\nSOURCE 1:\n{a.page_content}\nSOURCE 2:\n{b.page_content}')
    if r and ok_q(r.get("question", "")) and r.get("answer"):
        return {"question": str(r["question"]).strip(), "ground_truth": str(r["answer"]).strip(), "type": "multi_hop"}

def make_unanswerable(n):
    out, tries = [], 0
    while len(out) < n and tries < 6:
        tries += 1
        sample = "\n".join(chunks[i].page_content[:150] for i in random.sample(good, 8))
        r = gen(f"Given these excerpts from a book, write {n * 2} specific questions on the same topics that the book does NOT answer. {RULES}\n"
                f'Return {{"questions": [...]}}\n{sample}') or {}
        for q in r.get("questions", []):
            if ok_q(q) and len(out) < n:
                out.append({"question": str(q).strip(), "ground_truth": C.REFUSAL, "type": "unanswerable"})
    return out

rows = [json.loads(l) for l in open("data/questions.jsonl", encoding="utf-8")]
target = collections.Counter(r["type"] for r in rows)
keep = [r for r in rows if r["type"] == "adversarial" or not BAD.search(r["question"])]
flagged = collections.Counter(r["type"] for r in rows if r not in keep)
print("Flagged as broken:", dict(flagged), "| keeping", len(keep), "of", len(rows))
kept = collections.Counter(r["type"] for r in keep)

with open("data/questions.before_clean.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

new = []
new += collect(target["factual"] - kept["factual"], make_factual, "factual")
new += collect(target["multi_hop"] - kept["multi_hop"], make_multi, "multi_hop")
new += make_unanswerable(target["unanswerable"] - kept["unanswerable"])
final = keep + new
print(f"Final dataset: {len(final)} questions", dict(collections.Counter(r['type'] for r in final)))

with open("data/questions.jsonl", "w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

client = Client()
if client.has_dataset(dataset_name=NEW_DATASET):
    client.delete_dataset(dataset_name=NEW_DATASET)
ds = client.create_dataset(NEW_DATASET, description="Cleaned RAG eval questions (self-contained)")
client.create_examples(
    dataset_id=ds.id,
    inputs=[{"question": r["question"]} for r in final],
    outputs=[{"ground_truth": r["ground_truth"]} for r in final],
    metadata=[{"type": r["type"]} for r in final],
)
print(f"Uploaded {len(final)} examples to LangSmith dataset {NEW_DATASET}")
print("\nSample of NEW questions:")
for r in new[:6]:
    print(" -", r["type"], "|", r["question"])
