"""Build a markdown results table from the LangSmith experiments on the clean dataset."""
import collections, statistics, warnings
warnings.filterwarnings("ignore")
from langsmith import Client
from src import config as C

c = Client()
DS = C.DATASET_NAME
METRICS = ["answer_correctness", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "guardrail_compliance"]
ex = {e.id: e for e in c.list_examples(dataset_name=DS)}
print(f"Dataset {DS}: {len(ex)} questions")

best = {}
for p in c.list_projects(reference_dataset_name=DS):
    if "-" not in p.name:
        continue
    strat = p.name.rsplit("-", 1)[0]
    runs = list(c.list_runs(project_id=p.id, is_root=True))
    if strat not in best or len(runs) > len(best[strat][1]):
        best[strat] = (p, runs)

rows = []
for strat, (p, runs) in sorted(best.items()):
    ids = [r.id for r in runs]
    fb = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in range(0, len(ids), 50):
        for f in c.list_feedback(run_ids=ids[i:i + 50]):
            if f.score is not None:
                fb[f.key][f.run_id].append(f.score)
    avg = {m: (statistics.mean(s for v in fb[m].values() for s in v) if fb[m] else None) for m in METRICS}
    refusals = sum(1 for r in runs
                   if (r.outputs or {}).get("refused")
                   and (ex.get(r.reference_example_id).metadata or {}).get("type") in ("factual", "multi_hop")
                   if ex.get(r.reference_example_id))
    lat = [(r.end_time - r.start_time).total_seconds() for r in runs if r.end_time and r.start_time]
    rows.append((strat.replace("rag-", ""), len(runs), avg, refusals, statistics.median(lat) if lat else None))

def f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"

head = "| Strategy | Runs | Correctness | Faithfulness | Relevancy | Ctx precision | Ctx recall | Guardrail | Refused (answerable) | Median latency (s) |"
lines = [head, "|" + "---|" * 10]
for name, n, avg, refused, lat in rows:
    lines.append(f"| {name} | {n} | " + " | ".join(f(avg[m]) for m in METRICS) + f" | {refused} | {f(lat, 1)} |")
table = "\n".join(lines)
print("\n" + table)
open("RESULTS.md", "w", encoding="utf-8").write(table + "\n")
print("\nSaved RESULTS.md")
