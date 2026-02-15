#!/usr/bin/env python3
import importlib.util
spec = importlib.util.spec_from_file_location("t", "/projects/prjs1800/msc-thesis/01-arag-reproduction/scripts/05_flashrag_tools_scaffold.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tools = m.build_tools_for_arag("/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl", "/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_id_offset.json", "/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_fts.db", "/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index")
from arag.core.context import AgentContext
ctx = AgentContext()
print("KW:\n", tools.execute("keyword_search", ctx, keywords=["Albert Einstein"], top_k=2)[0][:600], "\n")
print("SEM:\n", tools.execute("semantic_search", ctx, query="Who developed the theory of relativity?", top_k=2)[0][:600], "\n")
print("READ:\n", tools.execute("read_chunk", ctx, chunk_ids=["0"])[0][:600], "\n")
print("OK logs:", ctx.get_summary()["retrieval_logs"][:2])