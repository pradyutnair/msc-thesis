"""Log Day 1 and Day 2 experiment results to Weights & Biases."""
import os
import wandb

# Load env
from dotenv import load_dotenv
load_dotenv("/projects/prjs1800/msc-thesis/.env")

PROJECT = "msc-thesis-rag-incremental"

# ── Day 1: Standard RAG ─────────────────────────────────────────────────────

experiments = [
    {
        "name": "day1_standard_rag_hotpotqa",
        "config": {
            "day": 1,
            "method": "Standard RAG",
            "dataset": "hotpotqa",
            "retriever": "E5-base-v2",
            "retrieval_topk": 5,
            "reranker": None,
            "refiner": None,
            "generator": "Qwen2.5-7B-Instruct",
            "framework": "vllm",
            "gpu": "A100-SXM4-40GB",
            "index": "e5_Flat (61GB, 21M passages)",
        },
        "metrics": {
            "em": 0.3164078325455773,
            "f1": 0.42010481255506066,
            "precision": 0.4418653652708935,
            "recall": 0.43178543584856927,
            # Retrieval quality
            "retrieval_recall_avg": 0.5003,
            "retrieval_recall_full": 0.239,
            "retrieval_recall_zero": 0.238,
            # Timing
            "retrieval_time_s": 272.6,
            "generation_time_s": 550.1,
            "total_time_s": 822.7,
            "n_examples": 7405,
        },
    },
    {
        "name": "day1_standard_rag_musique",
        "config": {
            "day": 1,
            "method": "Standard RAG",
            "dataset": "musique",
            "retriever": "E5-base-v2",
            "retrieval_topk": 5,
            "reranker": None,
            "refiner": None,
            "generator": "Qwen2.5-7B-Instruct",
            "framework": "vllm",
            "gpu": "A100-SXM4-40GB",
            "index": "e5_Flat (61GB, 21M passages)",
        },
        "metrics": {
            "em": 0.063301613570542,
            "f1": 0.13034746134810118,
            "precision": 0.13590550893897343,
            "recall": 0.14905678034556766,
            # Retrieval quality
            "retrieval_recall_avg": 0.2143,
            "retrieval_recall_full": 0.033,
            "retrieval_recall_zero": 0.558,
            # Per-hop recall
            "retrieval_recall_hop1": 0.335,
            "retrieval_recall_hop2": 0.116,
            "retrieval_recall_hop3": 0.065,
            "retrieval_recall_hop4": 0.032,
            # Timing
            "retrieval_time_s": 94.8,
            "generation_time_s": 169.4,
            "total_time_s": 264.2,
            "n_examples": 2417,
        },
    },
    # ── Day 2: Standard RAG + Reranker ───────────────────────────────────────
    {
        "name": "day2_reranker_rag_hotpotqa",
        "config": {
            "day": 2,
            "method": "Standard RAG + Reranker",
            "dataset": "hotpotqa",
            "retriever": "E5-base-v2",
            "retrieval_topk": 20,
            "reranker": "BGE-reranker-v2-m3",
            "rerank_topk": 5,
            "refiner": None,
            "generator": "Qwen2.5-7B-Instruct",
            "framework": "vllm",
            "gpu": "A100-SXM4-40GB",
            "index": "e5_Flat (61GB, 21M passages)",
        },
        "metrics": {
            "em": 0.3640783254557731,
            "f1": 0.47415745606978443,
            "precision": 0.49629296613779916,
            "recall": 0.48695820890790503,
            # Retrieval quality (after reranking)
            "retrieval_recall_avg": 0.577,
            "retrieval_recall_full": 0.336,
            "retrieval_recall_zero": 0.182,
            # Timing
            "retrieval_rerank_time_s": 1809.9,
            "generation_time_s": 364.3,
            "total_time_s": 2174.2,
            "n_examples": 7405,
        },
    },
    {
        "name": "day2_reranker_rag_musique",
        "config": {
            "day": 2,
            "method": "Standard RAG + Reranker",
            "dataset": "musique",
            "retriever": "E5-base-v2",
            "retrieval_topk": 20,
            "reranker": "BGE-reranker-v2-m3",
            "rerank_topk": 5,
            "refiner": None,
            "generator": "Qwen2.5-7B-Instruct",
            "framework": "vllm",
            "gpu": "A100-SXM4-40GB",
            "index": "e5_Flat (61GB, 21M passages)",
        },
        "metrics": {
            "em": 0.07695490277203144,
            "f1": 0.1552434337903806,
            "precision": 0.16498055506135262,
            "recall": 0.17113255314537876,
            # Retrieval quality (after reranking)
            "retrieval_recall_avg": 0.262,
            "retrieval_recall_full": 0.051,
            "retrieval_recall_zero": 0.465,
            # Per-hop recall (after reranking)
            "retrieval_recall_hop1": 0.396,
            "retrieval_recall_hop2": 0.148,
            "retrieval_recall_hop3": 0.124,
            "retrieval_recall_hop4": 0.047,
            # Timing
            "retrieval_rerank_time_s": 574.0,
            "generation_time_s": 117.0,
            "total_time_s": 691.0,
            "n_examples": 2417,
        },
    },
]

for exp in experiments:
    print(f"Logging: {exp['name']}")
    run = wandb.init(
        project=PROJECT,
        name=exp["name"],
        config=exp["config"],
        tags=[f"day{exp['config']['day']}", exp["config"]["dataset"], exp["config"]["method"]],
    )
    wandb.log(exp["metrics"])
    wandb.finish()

print("\nAll experiments logged to wandb project:", PROJECT)
