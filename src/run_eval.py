"""Runs 3 strategies x 100 questions x 6 metrics. Each strategy = one LangSmith experiment."""
import os, random
from langsmith import Client, evaluate
from . import config as C
from .pipeline import run_rag
from .strategies import STRATEGIES
from .evaluators import ALL

def get_data():
    if not C.DEV_N:
        return C.DATASET_NAME
    ex = list(Client().list_examples(dataset_name=C.DATASET_NAME))
    random.seed(1)
    return random.sample(ex, C.DEV_N)

def main():
    data = get_data()
    only = os.getenv("STRATEGY")   # e.g. STRATEGY=dense to run one strategy per day/session
    for name in STRATEGIES:
        if only and name != only:
            continue
        def target(inputs: dict, _name=name) -> dict:
            return run_rag(inputs["question"], _name)
        evaluate(
            target,
            data=data,
            evaluators=ALL,
            experiment_prefix=f"rag-{name}",
            metadata={"strategy": name, "top_k": C.TOP_K, "gen_model": C.GEN_MODEL, "judge": C.JUDGE_MODEL},
            max_concurrency=1,
        )
    print("Done. Compare the 3 experiments in LangSmith -> Datasets -> rag-eval-100 -> Experiments.")

if __name__ == "__main__":
    main()
